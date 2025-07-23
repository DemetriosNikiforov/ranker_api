"""
Pydantic модели для API ранжирования услуг.

Этот модуль содержит модели данных для валидации входящих запросов
и формирования структурированных ответов API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


class TextInput(BaseModel):
    """
    Модель для одиночного текстового запроса.

    Attributes:
        text: Текстовый запрос для ранжирования услуг
        top_K: Количество возвращаемых результатов
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Текстовый запрос для поиска и ранжирования услуг",
        example="Анализ крови на глюкозу",
    )

    top_k: int = Field(
        default=5,
        description="Топ k отранжированных элементов",
        example=5,
    )

    @field_validator("text", mode="before")
    def validate_text(cls, v):
        """Валидация текстового поля."""
        if not v or not v.strip():
            raise ValueError("Текст не может быть пустым")
        return v.strip()


class MultipleTextInput(BaseModel):
    """
    Модель для множественных текстовых запросов.

    Attributes:
        texts: Список текстовых запросов для ранжирования
        top_K: Количество возвращаемых результатов.

    """

    texts: List[str] = Field(
        ...,
        min_items=1,
        description="Список текстовых запросов для пакетной обработки",
        example=["Анализ крови на глюкозу", "УЗИ сердца", "Рентген легких"],
    )

    top_k: int = Field(
        default=5,
        description="Топ k отранжированных элементов",
        example=5,
    )

    @field_validator("texts", mode="before")
    def validate_texts(cls, v):
        """Валидация списка текстов."""
        if not v:
            raise ValueError("Список текстов не может быть пустым")

        validated_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Текст под индексом {i} не может быть пустым")
            if len(text.strip()) > 1000:
                raise ValueError(
                    f"Текст под индексом {i} превышает максимальную длину (1000 символов)"
                )
            validated_texts.append(text.strip())

        return validated_texts


class ServiceResult(BaseModel):
    """
    Модель результата ранжирования для одной услуги.

    Attributes:
        id: Уникальный идентификатор услуги
        service: Название услуги
        synonyms: Список синонимов для услуги
        score: Оценка релевантности (опционально)
    """

    id: int = Field(..., description="Уникальный идентификатор услуги", example=12345)

    service: str = Field(
        ...,
        description="Название услуги",
        example="Общий анализ крови с лейкоцитарной формулой",
    )

    synonyms: List[str] = Field(
        default_factory=list,
        description="Список синонимов и альтернативных названий услуги",
        example=["ОАК", "клинический анализ крови", "гемограмма"],
    )

    score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Оценка релевантности услуги к запросу (от 0 до 1)",
        example=0.95,
    )


class RankingResponse(BaseModel):
    """
    Модель ответа API для ранжирования услуг.

    Attributes:
        query: Исходный поисковый запрос
        results: Список ранжированных услуг
        total_found: Общее количество найденных услуг
        processing_time_ms: Время ранжирования
        error: Сообщение об ошибке (если есть)
    """

    query: str = Field(
        ..., description="Исходный поисковый запрос", example="Анализ крови на глюкозу"
    )

    results: List[ServiceResult] = Field(
        default_factory=list,
        description="Список ранжированных услуг, отсортированных по релевантности",
        example=[
            {
                "id": 12345,
                "service": "Глюкоза в сыворотке крови",
                "synonyms": ["сахар крови", "анализ на сахар"],
                "score": 0.95,
            }
        ],
    )

    total_found: int = Field(
        ..., ge=0, description="Общее количество найденных услуг", example=5
    )
    processing_time_ms: float = Field(
        ..., description="Время ранжирования", example=0.1
    )

    error: Optional[str] = Field(
        None,
        description="Сообщение об ошибке, если обработка запроса не удалась",
        example=None,
    )


class ErrorResponse(BaseModel):
    """
    Модель для ответов с ошибками.

    Attributes:
        detail: Детальное описание ошибки
        error_code: Код ошибки (опционально)
    """

    detail: str = Field(
        ...,
        description="Детальное описание ошибки",
        example="Сервис временно недоступен",
    )

    error_code: Optional[str] = Field(
        None,
        description="Код ошибки для программной обработки",
        example="SERVICE_UNAVAILABLE",
    )


class HealthResponse(BaseModel):
    """
    Модель ответа для проверки здоровья сервиса.

    Attributes:
        status: Статус сервиса
        service: Название сервиса
        version: Версия API
        models_loaded: Флаг загрузки моделей
    """

    status: str = Field(..., description="Статус сервиса", example="healthy")

    service: str = Field(
        ..., description="Название сервиса", example="Service Ranking API"
    )

    version: str = Field(..., description="Версия API", example="1.0.0")

    models_loaded: bool = Field(
        ...,
        description="Флаг успешной загрузки моделей машинного обучения",
        example=True,
    )
