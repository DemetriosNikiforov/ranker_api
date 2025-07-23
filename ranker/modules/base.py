"""
Предоставляет абстрактный базовый класс `BaseModel` и его реализацию `HFClassifier` для задачи классификации текста.
"""

from abc import ABC, abstractmethod
from typing import Any

from transformers import pipeline
import torch

from ranker.utils.load_models import ModelDownloader


class BaseModel(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError("Method load_model is not implemented")

    @abstractmethod
    def predict(self, text: str) -> Any:
        raise NotImplementedError("Method predict is not implemented")

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError("The method removes the hf model from memory")


class HFClassifier(BaseModel):
    """
    Реализация класса `BaseModel` для работы с текстовыми классификаторами Hugging Face.

    Использует библиотеку `transformers` для создания пайплайна классификации текста.
    Поддерживает автоматическое скачивание моделей через `ModelDownloader`.

    Атрибуты:
        pipeline: Объект пайплайна Hugging Face для классификации текста.
    """

    def __init__(self, model: str, hf_fallback: str = None):
        """
        Инициализация классификатора Hugging Face.

        Args:
            model (str): Путь к локальной модели или имя модели из Hugging Face.
            hf_fallback (str, optional): Резервное имя модели из Hugging Face для скачивания.
                Если модель не найдена локально, она будет скачана. Defaults to None.
        """
        self.hf_fallback = hf_fallback

        if hf_fallback:
            ModelDownloader.ensure_classification_model(model, hf_fallback)

        super().__init__(model)
        self.load_model()

    def load_model(self):
        """
        Загружает модель Hugging Face в пайплайн классификации текста.

        Использует GPU (CUDA) для ускорения вычислений.
        """
        self.pipeline = pipeline(
            task="text-classification", model=self.model, device="cuda"
        )

    def predict(self, text: str) -> int:
        """
        Выполняет классификацию текста с использованием модели Hugging Face.

        Args:
            text (str): Текст для классификации.

        Returns:
            int: Возращяет id типа услуги.
        """
        result = self.pipeline(text)
        return int(result[0]["label"])

    def unload(self):
        """Очистка памяти от hf модели"""
        self.pipeline = None
        torch.cuda.empty_cache()
