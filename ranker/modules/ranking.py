"""
Модуль реализации классов для текстового сопоставления и ранжирования с использованием
Hugging Face Transformers и Sentence Transformers.

Содержит:
- `MatchingClassifier` - базовый класс для бинарного классификатора текстовых пар
- `Ranker` - класс для ранжирования сервисов на основе семантической схожести и
релевантности, использующий биэнкодер и кроссэнкодер модели

Основные функции:
- Семантическое сравнение текстовых пар
- Ранжирование сервисов по релевантности
- Обработка синонимов и построение справочника
- Интеграция с локальными и облачными моделями Hugging Face
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ranker.const import (
    MODELS_BIENCODER_HF,
    MODELS_CROSSENCODER_HF,
    NAME_REFERENCE_COL,
    ROOT_PATH_MODEL,
    MODELS_FOLDER,
)

from ranker.utils.load_models import ModelDownloader

MODEL_RANKER = os.path.join(ROOT_PATH_MODEL, MODELS_FOLDER, "ranker")
MODEL_SENTENCE = os.path.join(ROOT_PATH_MODEL, MODELS_FOLDER, "bi_encoder_ranker")
REFERENCE_DATA = os.path.join(ROOT_PATH_MODEL, "data_search", "reference.csv")
SYNONYMS = os.path.join(ROOT_PATH_MODEL, "data_search", "search.json")


class MatchingClassifier:
    """
    Базовый класс для бинарного классификатора текстовых пар.

    Предназначен для определения схожести/релевантности двух текстовых фрагментов.
    Использует кроссэнкодерную модель Hugging Face для классификации пар текста.

    Атрибуты:
        model: Загруженная модель Hugging Face для последовательной классификации
        tokenizer: Токенизатор, соответствующий модели
    """

    def __init__(self, model: str, hf_fallback: str = None):
        self.model = model
        self.hf_fallback = hf_fallback

        if hf_fallback:
            ModelDownloader.ensure_classification_model(model, hf_fallback)

        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model).to(
            "cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        return model, tokenizer

    def predict(self, text_1, text_2):
        inputs = self.tokenizer(
            text_1,
            text_2,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        scores = probs[:, 1]
        labels = torch.argmax(probs, dim=1).cpu().tolist()
        scores = scores.cpu().tolist()

        return labels, scores

    def unload(self):
        """Очистка памяти от hf модели"""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()


class Ranker(MatchingClassifier):
    """
    Класс для ранжирования сервисов на основе семантической схожести и релевантности.

    Расширяет функциональность MatchingClassifier, добавляя:
    - Биэнкодерный подход для предварительного отбора кандидатов
    - Ранжирование на основе кроссэнкодерной модели
    - Поддержку синонимов и справочника сервисов

    Атрибуты:
        bi_encoder: Биэнкодерная модель для вычисления эмбеддингов
        reference: Справочник сервисов в формате DataFrame
        synonym_lookup: Словарь синонимов для сервисов
    """

    def __init__(self):
        """
        Инициализирует ранжировщик с предопределенными параметрами.

        Использует:
            - Кроссэнкодер: const.MODEL_RANKER / const.MODELS_CROSSENCODER_HF
            - Биэнкодер: const.MODEL_SENTENCE / const.MODELS_BIENCODER_HF
        """

        super().__init__(model=MODEL_RANKER, hf_fallback=MODELS_CROSSENCODER_HF)

        ModelDownloader.ensure_sentence_transformer(
            local_path=MODEL_SENTENCE, hf_repo=MODELS_BIENCODER_HF
        )
        self.bi_encoder = SentenceTransformer(MODEL_SENTENCE)

        self.reference = self._load_data(REFERENCE_DATA)
        self.guide_data = self._load_json(SYNONYMS)
        self.synonym_lookup = self._build_synonym_lookup(self.guide_data)

    def _load_data(self, path) -> pd.DataFrame:
        """
        Загружает справочные данные из CSV файла.

        Args:
            path (str): Путь к CSV файлу с данными

        Returns:
            pd.DataFrame: Загруженные данные в формате DataFrame
        """
        return pd.read_csv(path)

    def _load_json(self, path) -> List[Dict[str, Any]]:
        """
        Загружает JSON файл с синонимами.

        Args:
            path (str): Путь к JSON файлу с данными

        Returns:
            List[Dict[str, Any]]: Список словарей с данными из JSON
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_synonym_lookup(self, data: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Создает словарь синонимов для быстрого поиска.

        Args:
            data (List[Dict[str, Any]]): Список словарей с данными о синонимах

        Returns:
            Dict[int, List[str]]: Словарь соответствий {id: [синонимы]}
        """

        return {
            item["id"]: list(set(item["synonyms"]))[:5]
            for item in data
            if "synonyms" in item
        }

    def get_top_k_relevant_samples(self, query, filtered_data, k=100):
        """
        Находит топ-K наиболее релевантных образцов для запроса.

        Использует биэнкодерную модель для вычисления косинусного сходства.

        Args:
            query (str): Текстовый запрос для поиска
            filtered_data (pd.DataFrame): DataFrame с кандидатами для ранжирования
            k (int, optional): Количество возвращаемых результатов. Defaults to 100.

        Returns:
            pd.DataFrame: DataFrame с топ-K релевантными образцами и их оценками
        """

        if filtered_data is None or filtered_data.empty:
            print(
                f"[INFO] Пропущено ранжирование: пустой filtered_data для запроса: {query}"
            )
            return pd.DataFrame()

        query_embedding = self.bi_encoder.encode(
            query, convert_to_numpy=True, device="cuda:0"
        )

        embeddings_filtered = self.bi_encoder.encode(
            filtered_data[NAME_REFERENCE_COL].values,
            convert_to_numpy=True,
            device="cuda:0",
        )

        similarity_scores = util.pytorch_cos_sim(query_embedding, embeddings_filtered)[
            0
        ]

        top_k = torch.topk(similarity_scores, k=min(k, len(filtered_data)))
        top_k_idx = top_k.indices.cpu().numpy()
        top_k_scores = top_k.values.cpu().numpy()

        top_k_df = filtered_data.iloc[top_k_idx].copy()
        top_k_df["score"] = top_k_scores

        return top_k_df.reset_index(drop=True)

    def collect_ranked_services(
        self, services_names: List[str], filtered_data: pd.DataFrame, top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Собирает и ранжирует сервисы по релевантности для каждого запроса.

        Args:
            services_names (List[str]): Список текстовых запросов для ранжирования
            filtered_data (pd.DataFrame): DataFrame с кандидатами для ранжирования
            top_k (int, optional): Количество возвращаемых результатов для каждого запроса.
                Defaults to 5.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Словарь с результатами ранжирования в формате:
                {
                    "запрос": [
                        {
                            "id": int,
                            "service": str,
                            "synonyms": List[str],
                            "score": float
                        },
                        ...
                    ],
                    ...
                }
        """
        results = {}

        for service_name in services_names:
            print(f"Processing service: {service_name}")

            filtered_subset = self.get_top_k_relevant_samples(
                service_name, filtered_data
            )

            if filtered_subset.empty:
                print(
                    f"[INFO] Пропускаем ранжирование — нет подходящих сервисов для: {service_name}"
                )
                continue

            predictions = []

            for filtered_id, reference_service in tqdm(
                zip(
                    filtered_subset["local_id"],
                    filtered_subset[NAME_REFERENCE_COL],
                ),
                desc=f"Ранжирование: {service_name}",
                total=len(filtered_subset),
            ):
                _, ranker_score = self.predict(service_name, reference_service)
                predictions.append((filtered_id, reference_service, ranker_score))

            if not predictions:
                continue

            predictions.sort(key=lambda x: x[2], reverse=True)
            top_k_preds = predictions[:top_k]

            results[service_name] = [
                {
                    "id": p[0],
                    "service": p[1],
                    "synonyms": self.synonym_lookup.get(p[0], []),
                    "score": p[2][0],
                }
                for p in top_k_preds
            ]

        return results
