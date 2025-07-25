from pathlib import Path
import os

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine

DB_CONFIG_PATH = ".env"
ROOT_SQL_DIR = "queries/shop_budu"


class PGManager:
    """
    Класс для работы с базой данных Postgres.
    Использует конфигурацию из файла .env.
    """

    def __init__(self):
        config = dotenv_values(DB_CONFIG_PATH)
        self.root_sql_dir = ROOT_SQL_DIR

        self.user = config.get("PG_USER")
        self.host = config.get("PG_HOST")
        self.password = config.get("PG_PASSWORD")
        self.port = config.get("PG_PORT")
        self.dbname = config.get("PG_DBNAME")
        self.engine = self.create_engine()

    def create_engine(self):
        """Создает и возвращает соединение с базой данных."""
        db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        engine = create_engine(db_url)
        return engine

    def get_data(self, sql, **params):
        """
        Выполняет SQL запрос и возвращает результат в виде DataFrame.

        sql: str or path to file
        params: dict with values to format sql

        return: pd.DataFrame
        """
        if os.path.exists(sql):
            with open(sql, "r") as file:
                sql = file.read().format(**params)
            return pd.read_sql(sql, con=self.engine)
        elif isinstance(sql, str):
            return pd.read_sql(sql, con=self.engine)
        else:
            raise ValueError("SQL должен быть строкой или путем к файлу.")

    def get_guide_base_list(self, sql="get_guide_base_list.sql"):
        """Возвращает список всех guide_base таблиц с их id и названием."""
        sql_query = open(os.path.join(self.root_sql_dir, sql), "r").read()
        return self.get_data(sql_query)

    def get_guide_base_by_id(self, gb_id, sql="get_guide_base_by_id.sql"):
        """Возвращает данные прайса из guide_base таблицы по id."""
        sql_query = (
            open(os.path.join(self.root_sql_dir, sql), "r")
            .read()
            .format(guide_base_id=f"guide_base_{gb_id}")
        )
        return self.get_data(sql_query)

    def get_guide_standart(self, sql="get_guide_standart.sql"):
        """Возвращает эталонный справочник (guide_standart)."""
        return self.get_data(os.path.join(self.root_sql_dir, sql))

    def get_actual_site_service_list(self, sql="get_actual_site_service_list.sql"):
        """Возвращает актуальный список услуг из маркетплэйса."""
        service_list = self.get_data(os.path.join(self.root_sql_dir, sql))
        service_list["sale_count"] = service_list["sale_count"].fillna(0).astype(int)

        return service_list

    def get_synonyms_for_guide_standart(self, sql="get_synonim_for_gs.sql"):
        """Возвращает список кастомных синонимов для model_type == \App\Models\GuideStandard."""
        return self.get_data(os.path.join(self.root_sql_dir, sql))

    def get_all_matching_services(self):
        """Возвращает все смэтченные услуги из всех guide_base таблиц."""
        guide_base_ids = self.get_guide_base_list().guide_base_id.values
        excluded_ids = {55}  # legacy price for synonym
        all_services = []

        for guide_base_id in guide_base_ids:
            if guide_base_id in excluded_ids:
                continue

            try:
                guide_data = self.get_guide_base_by_id(guide_base_id)
                if not guide_data.empty and not guide_data.isna().all().all():
                    all_services.append(guide_data)
            except Exception as e:
                print(f"Error processing guide_base_id {guide_base_id}: {e}")
                continue

        return (
            pd.concat(all_services, ignore_index=True)
            if all_services
            else pd.DataFrame()
        )

    @staticmethod
    def get_service_names_from_prices(all_services_df, service_name):
        """Возвращает уникальные названия услуги из других прайсов."""
        service_names = all_services_df.loc[
            all_services_df["local_name"] == service_name, "service_name"
        ]
        return service_names.drop_duplicates().tolist()

    @staticmethod
    def merge_custom_synonyms(reference_df: pd.DataFrame, synonym_df: pd.DataFrame):
        """Объединяет DataFrame с кастомными синонимами."""
        synonym_map = synonym_df.groupby("local_name")["synonym"].apply(list).to_dict()

        reference_df["synonyms"] += reference_df["name"].apply(
            lambda name: synonym_map.get(name, [])
        )
        return reference_df

    def get_search_raw_data(self, save_path="search.json"):
        """Возвращает данные для работы с поиском"""
        reference_df = self.get_actual_site_service_list()
        all_services = self.get_all_matching_services()
        synonym_df = self.get_synonyms_for_guide_standart()

        # Добавляем названия услуг из прайсов в справочник
        reference_df["synonyms"] = reference_df["name"].apply(
            lambda x: self.get_service_names_from_prices(all_services, x)
        )

        # Объединяем с кастомными синонимами
        reference_df = self.merge_custom_synonyms(reference_df, synonym_df)

        reference_df.to_json(save_path, orient="records", indent=4, force_ascii=False)
        return reference_df

    def get_guide_type_list(self, sql="get_guide_type_list.sql"):
        """Возвращает список  guide_type."""
        return self.get_data(os.path.join(self.root_sql_dir, sql))
