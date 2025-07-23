import os

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelDownloader:
    @staticmethod
    def ensure_classification_model(local_path: str, hf_repo: str):
        if not os.path.exists(local_path):
            print(f"Создаём директорию: {local_path}")
            os.makedirs(local_path, exist_ok=True)

        if not os.listdir(local_path):
            print(f"Скачивается модель: '{hf_repo}' → '{local_path}'")
            tokenizer = AutoTokenizer.from_pretrained(hf_repo)
            model = AutoModelForSequenceClassification.from_pretrained(hf_repo)
            
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)

    @staticmethod
    def ensure_sentence_transformer(local_path: str, hf_repo: str):
        if not os.path.exists(local_path):
            print(f"Создаём директорию: {local_path}")
            os.makedirs(local_path, exist_ok=True)

        if not os.listdir(local_path):
            print(f"Скачиваем SentenceTransformer '{hf_repo}' → '{local_path}'")
            model = SentenceTransformer(hf_repo)
            model.save(local_path)
