"""
Сервисный слой для ранжирования услуг.

Этот модуль содержит бизнес-логику для обработки запросов ранжирования,
инкапсулирует работу с моделями машинного обучения и предоставляет
упрощенный интерфейс для API.
"""

import logging
from typing import List, Dict, Any
import gc
import torch

from models import ServiceResult

# Импорт из исходных файлов
from ranker.matching_baskets import CollectingBaskets

logger = logging.getLogger(__name__)


class RankingService:
    """
    Сервис для ранжирования услуг на основе текстовых запросов.

    Этот класс инкапсулирует логику работы с моделями машинного обучения
    для классификации и ранжирования услуг. Он предоставляет упрощенный
    интерфейс для работы с системой ранжирования.
    """

    def __init__(self):
        """
        Инициализация сервиса ранжирования.

        Загружает необходимые модели машинного обучения и подготавливает
        систему к обработке запросов.

        Raises:
            ModelLoadError: При ошибке загрузки моделей
        """
        logger.info("Инициализация RankingService...")

        try:
            # Инициализация системы сбора корзин (CollectingBaskets)
            self.collecting_baskets = CollectingBaskets()
            logger.info("CollectingBaskets успешно инициализирован")

            self._initialized = True
            logger.info("RankingService успешно инициализирован")

        except Exception as e:
            logger.error(f"Ошибка при инициализации RankingService: {e}")

    def is_initialized(self) -> bool:
        """
        Проверка состояния инициализации сервиса.

        Returns:
            bool: True, если сервис инициализирован и готов к работе
        """
        return getattr(self, "_initialized", False)

    def rank_services(self, query: str, top_k: int = 5) -> List[ServiceResult]:
        """
        Ранжирование услуг для заданного запроса.

        Этот метод выполняет полный цикл обработки запроса:
        1. Классификация запроса по категориям
        2. Фильтрация услуг по категории
        3. Семантическое ранжирование с помощью bi-encoder
        4. Точное ранжирование с помощью cross-encoder

        Args:
            query: Текстовый запрос для поиска услуг
            top_k: Максимальное количество возвращаемых результатов

        Returns:
            List[ServiceResult]: Список ранжированных услуг

        Raises:
            ServiceNotFoundError: Если услуги не найдены
            Exception: При других ошибках обработки
        """
        if not self.is_initialized():
            raise Exception("Сервис не инициализирован")

        if not query or not query.strip():
            raise ValueError("Запрос не может быть пустым")

        query = query.strip()
        logger.info(f"Начало ранжирования для запроса: '{query[:100]}...'")

        try:
            # Используем метод second_rank_services для получения результатов
            ranking_results = self.collecting_baskets.second_rank_services(query, top_k)

            if not ranking_results:
                logger.warning(f"Услуги не найдены для запроса: '{query}'")

            # Извлекаем результаты для запроса
            services_data = ranking_results.get(query, [])

            if not services_data:
                logger.warning(f"Пустой результат ранжирования для запроса: '{query}'")

            # Ограничиваем количество результатов
            services_data = services_data[:top_k]

            # Преобразуем результаты в объекты ServiceResult
            results = []
            for service_info in services_data:
                try:
                    service_result = ServiceResult(
                        id=service_info.get("id", 0),
                        service=service_info.get("service", ""),
                        synonyms=service_info.get("synonyms", []),
                        score=service_info.get("score", -1),
                    )
                    results.append(service_result)

                except Exception as e:
                    logger.warning(
                        f"Ошибка при обработке результата {service_info}: {e}"
                    )
                    continue

            if not results:
                logger.warning(
                    f"Не удалось обработать результаты для запроса: '{query}'"
                )

            logger.info(f"Успешно найдено {len(results)} услуг для запроса: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Ошибка при ранжировании услуг для запроса '{query}': {e}")
            raise Exception(f"Внутренняя ошибка при ранжировании: {e}")

    def get_service_categories(self) -> List[str]:
        """
        Получение списка доступных категорий услуг.

        Returns:
            List[str]: Список категорий услуг
        """
        if not self.is_initialized():
            raise Exception("Сервис не инициализирован")

        try:
            # Получаем уникальные категории из справочных данных
            categories = self.collecting_baskets.ranker.reference.columns.tolist()
            return [cat for cat in categories if "category" in cat.lower()]

        except Exception as e:
            logger.error(f"Ошибка при получении категорий: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по загруженным данным.

        Returns:
            Dict[str, Any]: Статистика по услугам и категориям
        """
        if not self.is_initialized():
            return {"error": "Сервис не инициализирован"}

        try:
            reference_df = self.collecting_baskets.ranker.reference

            stats = {
                "total_services": len(reference_df),
                "total_synonyms": len(self.collecting_baskets.ranker.synonym_lookup),
                "data_columns": reference_df.columns.tolist(),
                "memory_usage_mb": reference_df.memory_usage(deep=True).sum()
                / 1024
                / 1024,
            }

            return stats

        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """
        Освобождение ресурсов и очистка памяти.

        Этот метод должен вызываться при завершении работы приложения
        для корректного освобождения ресурсов GPU и оперативной памяти.
        """
        logger.info("Начало очистки ресурсов RankingService...")

        try:
            if hasattr(self, "collecting_baskets") and self.collecting_baskets:
                # Вызываем метод clear_memory для освобождения ресурсов
                self.collecting_baskets.clear_memory()
                self.collecting_baskets = None
                logger.info("CollectingBaskets ресурсы освобождены")

            # Принудительная очистка памяти
            gc.collect()

            # Очистка CUDA кэша, если доступно
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA кэш очищен")

            self._initialized = False
            logger.info("RankingService ресурсы успешно освобождены")

        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов: {e}")

    def __del__(self):
        """
        Деструктор для автоматической очистки ресурсов.
        """
        try:
            if getattr(self, "_initialized", False):
                self.cleanup()
        except Exception as e:
            logger.error(f"Ошибка в деструкторе RankingService: {e}")
