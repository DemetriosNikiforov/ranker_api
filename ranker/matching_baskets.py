"""
Модуль реализации системы сбора и фильтрации сервисов на основе текстового запроса.

Содержит класс `CollectingBaskets`, который интегрирует классификаторы и ранжировщик
для последовательной обработки текстовых запросов и возврата релевантных сервисов.

Основные этапы обработки:
1. Классификация запроса на категорию и подкатегорию
2. Фильтрация справочника сервисов по классу
3. Предварительный отбор кандидатов через биэнкодер
4. Финальное ранжирование через кроссэнкодер
"""

from ranker.const import NAME_CATEGORIZER_COL, NAME_SUBCATEGORIZER_COL
from ranker.modules.classifiers import (
    LabsClassifier,
    MainClassifier,
    DiagnosticClassifier,
)
from ranker.modules.ranking import Ranker


class CollectingBaskets:
    """
    Класс для комплексной обработки текстовых запросов с целью сбора релевантных сервисов.

    Интегрирует несколько уровней классификации и двухэтапное ранжирование для
    обеспечения точного и контекстуально релевантного поиска сервисов.

    Атрибуты:
        classifier (MainClassifier): Основной классификатор категорий
        subclassifier (LabsClassifier): Классификатор подкатегорий лабораторных анализов
        subclassifier_diag (DiagnosticClassifier): Классификатор подкатегорий диагностик
        ranker (Ranker): Объект для ранжирования сервисов
    """

    def __init__(self):
        self.classifier = MainClassifier()
        self.subclassifier_analyzes = LabsClassifier()
        self.subclassifier_diagnostics = DiagnosticClassifier()
        self.ranker = Ranker()

    def _filted_data(self, column, value):
        """
        Фильтрует справочник сервисов по заданному столбцу и значению.

        Args:
            column (str): Имя колонки для фильтрации
            value (Any): Значение для фильтрации

        Returns:
            pd.DataFrame: Отфильтрованный DataFrame с сервисами
        """
        return self.ranker.reference[self.ranker.reference[column] == value]

    def process(self, text):
        """
        Обрабатывает текстовый запрос для определения категории и фильтрации сервисов.

        Выполняет последовательную классификацию:
        1. Основная категория
        2. Подкатегория (при необходимости для категорий 64[Анализы] и 55[Диагностика])

        Args:
            text (str): Текстовый запрос пользователя

        Returns:
            pd.DataFrame: Отфильтрованный DataFrame с сервисами для ранжирования
        """

        self.main_category = self.classifier.predict(text)
        if self.main_category == 64:  # Категория лабораторных анализов
            self.subcategory = self.subclassifier_analyzes.predict(text)

            return self._filted_data(
                column=NAME_SUBCATEGORIZER_COL, value=self.subcategory
            )
        elif self.main_category == 55:  # Категория диагностических процедур
            self.subcategory = self.subclassifier_diagnostics.predict(text)

            return self._filted_data(
                column=NAME_SUBCATEGORIZER_COL, value=self.subcategory
            )
        # Все другие категории
        return self._filted_data(column=NAME_CATEGORIZER_COL, value=self.main_category)

    def first_rank_services(self, query):
        """
        Выполняет предварительное ранжирование сервисов.

        Args:
            query (str): Текстовый запрос пользователя

        Returns:
            pd.DataFrame: Топ-K наиболее релевантных сервисов
        """
        filter_reference = self.process(query)

        return self.ranker.get_top_k_relevant_samples(query, filter_reference)

    def second_rank_services(self, query, top_k=5):
        """
        Выполняет финальное ранжирование сервисов с детальной оценкой.

        Args:
            query (str): Текстовый запрос пользователя
            top_k (int, optional): Количество возвращаемых результатов для каждого запроса.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Результаты ранжирования в формате:
                {
                    "запрос": [
                        {
                            "id": int,
                            "service": str,
                            "synonyms": List[str],
                            "score" : float
                        },
                        ...
                    ]
                }
        """

        filter_data = self.first_rank_services(query)

        return self.ranker.collect_ranked_services([query], filter_data, top_k)

    def clear_memory(self):
        """Очистка памяти"""
        self.subclassifier_analyzes.unload()
        self.subclassifier_diagnostics.unload()
        self.classifier.unload()
        self.ranker.unload()
