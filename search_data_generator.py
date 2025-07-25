import json
import re
import logging
import os

from tqdm import tqdm

from utils.db_manager import PGManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("search_data_generator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    try:
        os.makedirs("data/data_search/", exist_ok=True)

        # 1. Получение данных из БД
        logger.info("Инициализация подключения к базе данных...")
        db = PGManager()

        logger.info("Загрузка сырых данных для поиска...")
        db.get_search_raw_data("./data/data_search/search.json")
        logger.info("Данные успешно сохранены в data/data_search/search.json")

        # 2. Загрузка и обработка JSON
        logger.info("Чтение файла search.json...")
        with open("data/data_search/search.json", "r", encoding="utf-8") as f:
            search_data = json.load(f)
        logger.info(f"Загружено {len(search_data)} элементов")

        # 3. Сбор всех терминов и синонимов
        logger.info("Формирование списка терминов и синонимов...")
        all_items = [
            item["name"]
            for item in tqdm(search_data, desc="Обработка основных терминов")
        ] + [
            synonym
            for item in tqdm(search_data, desc="Обработка синонимов")
            for synonym in item.get("synonyms", [])
        ]
        logger.info(f"Всего собрано {len(all_items)} терминов")

        # 4. Нормализация текста
        def normalize_text(names):
            normalized = set()
            logger.info("Начало нормализации текста...")

            for name in tqdm(names, desc="Нормализация"):
                try:
                    # Удаляем спецсимволы, приводим к нижнему регистру
                    clean_name = re.sub(
                        r"[^a-zA-Zа-яА-Яα-ωΑ-Ω0-9Ёё]", " ", name.lower()
                    ).strip()
                    normalized.update(clean_name.split())
                except Exception as e:
                    logger.warning(f"Ошибка при обработке '{name}': {str(e)}")

            logger.info(f"Получено {len(normalized)} уникальных токенов")
            return normalized

        unique_terms = normalize_text(all_items)

        # 5. Сохранение результата
        logger.info("Формирование итогового словаря...")
        result_data = {term: 1 for term in tqdm(unique_terms, desc="Форматирование")}

        logger.info("Сохранение в terms.json...")
        with open("data/data_search/terms.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Готово! Успешно сохранено {len(result_data)} терминов")

        reference = db.get_data(
            """select DISTINCT ON (gbd.local_name) gbd.local_name, gbd.local_id, gbd.guide_type_id,
gbd.guide_type_name,gbd.guide_type_parent_id,gbd.guide_type_parent_name  from guide_base_data gbd  where gbd.local_id notnull"""
        )

        reference.to_csv("data/data_search/reference.csv", index=False)

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
