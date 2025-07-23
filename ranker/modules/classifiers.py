"""
Содержит классы для работы с предопределенными моделями классификации текста:
- MainClassifier - основной классификатор категорий
- LabsClassifier - классификатор подкатегорий лабораторных анализов
- DiagnosticClassifier - классификатор подкатегорий диагностических процедур

Использует преднастроенные пути к моделям и резервные названия из Hugging Face.
"""

import os

from ranker.const import (
    MODEL_CATEGORIZER_HF,
    MODEL_SUBCATEGORIZER_ANALYZES_HF,
    MODEL_SUBCATEGORIZER_DIAGNOSTICS_HF,
    ROOT_PATH_MODEL,
    MODELS_FOLDER,
)
from ranker.modules.base import HFClassifier

# Пути к локальным моделям
MODEL_CATEGORIZER = os.path.join(ROOT_PATH_MODEL, MODELS_FOLDER, "categorizer")
MODEL_SUBCATEGORIZER_LABS = os.path.join(
    ROOT_PATH_MODEL, MODELS_FOLDER, "subcategorizer_labs"
)
MODEL_SUBCATEGORIZER_DIAG = os.path.join(
    ROOT_PATH_MODEL, MODELS_FOLDER, "subcategorizer_diagnostic"
)


class MainClassifier(HFClassifier):
    """
    Основной классификатор для категоризации текстовых данных.

    Использует предопределенную модель для классификации текста по категориям.
    В случае отсутствия локальной модели автоматически загружает резервную модель
    из Hugging Face.
    """

    def __init__(self):
        """
        Инициализирует классификатор с предопределенными параметрами.

        Использует:
            - Локальную модель: const.ROOT_PATH_MODEL/categorizer
            - Резервную модель: const.MODEL_CATEGORIZER_HF
        """
        super().__init__(model=MODEL_CATEGORIZER, hf_fallback=MODEL_CATEGORIZER_HF)


class LabsClassifier(HFClassifier):
    """
    Классификатор для подкатегорий лабораторных анализов.

    Специализированный классификатор для анализа текстовых данных в области
    лабораторной диагностики. Обеспечивает автоматическое скачивание резервной
    модели при необходимости.
    """

    def __init__(self):
        """
        Инициализирует классификатор лабораторных анализов.

        Использует:
            - Локальную модель: const.ROOT_PATH_MODEL/subcategorizer_labs
            - Резервную модель: const.MODEL_SUBCATEGORIZER_ANALYZES_HF
        """
        super().__init__(
            model=MODEL_SUBCATEGORIZER_LABS,
            hf_fallback=MODEL_SUBCATEGORIZER_ANALYZES_HF,
        )


class DiagnosticClassifier(HFClassifier):
    """
    Классификатор для подкатегорий диагностических процедур.

    Предназначен для классификации текстовых данных в области диагностических
    исследований. Обеспечивает отказоустойчивую работу через резервную модель
    из Hugging Face.
    """

    def __init__(self):
        """
        Инициализирует классификатор диагностических процедур.

        Использует:
            - Локальную модель: const.ROOT_PATH_MODEL/subcategorizer_diagnostic
            - Резервную модель: const.MODEL_SUBCATEGORIZER_DIAGNOSTICS_HF
        """
        super().__init__(
            model=MODEL_SUBCATEGORIZER_DIAG,
            hf_fallback=MODEL_SUBCATEGORIZER_DIAGNOSTICS_HF,
        )
