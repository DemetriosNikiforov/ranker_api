"""
FastAPI приложение для ранжирования услуг.

Этот модуль предоставляет REST API для ранжирования услуг на основе текстового запроса.
API использует модели машинного обучения для классификации и ранжирования услуг.
"""

import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from models import RankingResponse, TextInput, MultipleTextInput
from service import RankingService

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для хранения сервиса ранжирования
ranking_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения.

    Инициализирует сервис ранжирования при запуске и корректно
    освобождает ресурсы при остановке приложения.
    """
    global ranking_service

    # Startup
    logger.info("Инициализация сервиса ранжирования...")
    try:
        ranking_service = RankingService()
        logger.info("Сервис ранжирования успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка при инициализации сервиса: {e}")

    yield

    # Shutdown
    logger.info("Освобождение ресурсов...")
    if ranking_service:
        ranking_service.cleanup()
    logger.info("Ресурсы освобождены")


# Создание FastAPI приложения
app = FastAPI(
    title="Service Ranking API",
    description="API для ранжирования услуг на основе текстовых запросов",
    version="1.0.0",
    lifespan=lifespan,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене следует ограничить список доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """
    Проверка работоспособности API.

    Returns:
        dict: Сообщение о статусе API
    """
    return {"message": "Service Ranking API работает"}


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Детальная проверка здоровья API.

    Returns:
        dict: Детальная информация о состоянии сервиса
    """
    global ranking_service

    if ranking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис ранжирования не инициализирован",
        )

    return {
        "status": "healthy",
        "service": "Service Ranking API",
        "version": "1.0.0",
        "models_loaded": True,
    }


@app.post("/rank", response_model=RankingResponse, tags=["Ranking"])
async def rank_single_text(input_data: TextInput) -> RankingResponse:
    """
    Ранжирование услуг для одного текстового запроса.

    Args:
        input_data: Объект с текстовым запросом

    Returns:
        RankingResponse: Результат ранжирования услуг

    Raises:
        HTTPException: При ошибках обработки запроса
    """
    global ranking_service

    if ranking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис ранжирования не доступен",
        )

    try:
        logger.info(f"Обработка запроса: {input_data.text[:100]}...")

        start = time.perf_counter()

        result = ranking_service.rank_services(input_data.text, input_data.top_k)

        end = time.perf_counter()

        logger.info(f"Запрос обработан успешно. Найдено услуг: {len(result)}")

        return RankingResponse(
            query=input_data.text,
            results=result,
            total_found=len(result),
            processing_time_ms=(end - start) * 1000,
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}",
        )


@app.post("/rank/batch", response_model=Dict[str, RankingResponse], tags=["Ranking"])
async def rank_multiple_texts(
    input_data: MultipleTextInput,
) -> Dict[str, RankingResponse]:
    """
    Ранжирование услуг для нескольких текстовых запросов.

    Args:
        input_data: Объект с массивом текстовых запросов

    Returns:
        Dict[str, RankingResponse]: Результаты ранжирования для каждого запроса

    Raises:
        HTTPException: При ошибках обработки запросов
    """
    global ranking_service

    if ranking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис ранжирования не доступен",
        )

    if len(input_data.texts) > 50:  # Ограничение на количество запросов
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Превышено максимальное количество запросов (50)",
        )

    results = {}

    try:
        logger.info(f"Обработка пакета из {len(input_data.texts)} запросов")

        for i, text in enumerate(input_data.texts):
            try:
                logger.info(
                    f"Обработка запроса {i+1}/{len(input_data.texts)}: {text[:50]}..."
                )

                start = time.perf_counter()

                result = ranking_service.rank_services(text, input_data.top_k)

                end = time.perf_counter()

                results[f"query_{i}"] = RankingResponse(
                    query=text,
                    results=result,
                    total_found=len(result),
                    processing_time_ms=(end - start) * 1000,
                )

            except Exception as e:
                logger.error(f"Ошибка при обработке запроса {i+1}: {e}")
                results[f"query_{i}"] = RankingResponse(
                    query=text,
                    results=[],
                    total_found=0,
                    processing_time_ms=0,
                    error=str(e),
                )

        logger.info(f"Пакетная обработка завершена")
        return results

    except Exception as e:
        logger.error(f"Критическая ошибка при пакетной обработке: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
