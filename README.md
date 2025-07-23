# Service Ranking API

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## 🛠 Технологический стек

- **Python 3.8+**
- **FastAPI** - современный веб-фреймворк
- **Pydantic** - валидация данных
- **PyTorch** - машинное обучение
- **Uvicorn** - ASGI сервер

## 📦 Установка

### Клонирование репозитория

```bash
git clone https://github.com/yourusername/service-ranking-api.git
cd service-ranking-api
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## 🚀 Быстрый старт

### Запуск сервера

```bash
python main.py
```

Или с помощью uvicorn:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

API будет доступно по адресу: `http://127.0.0.1:8000`

### Интерактивная документация

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## 📖 Использование API

### Проверка состояния

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### Ранжирование одного запроса

```bash
curl -X POST "http://127.0.0.1:8000/rank" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Анализ крови на глюкозу",
       "top_k": 5
     }'
```

### Пакетная обработка

```bash
curl -X POST "http://127.0.0.1:8000/rank/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Анализ крови на глюкозу",
         "УЗИ сердца",
         "Рентген легких"
       ],
       "top_k": 3
     }'
```

## 📝 Структура ответа

### Одиночный запрос

```json
{
  "query": "Анализ крови на глюкозу",
  "results": [
    {
      "id": 12345,
      "service": "Глюкоза в сыворотке крови",
      "synonyms": ["сахар крови", "анализ на сахар"],
      "score": 0.95
    }
  ],
  "total_found": 5,
  "processing_time_ms": 45.2,
  "error": null
}
```

### Пакетная обработка

```json
{
  "query_0": {
    "query": "Анализ крови на глюкозу",
    "results": [...],
    "total_found": 5,
    "processing_time_ms": 45.2
  },
  "query_1": {
    "query": "УЗИ сердца",
    "results": [...],
    "total_found": 3,
    "processing_time_ms": 38.7
  }
}
```

## 🏗 Архитектура проекта

```
service-ranking-api/
├── main.py              # FastAPI приложение и роуты
├── service.py           # Бизнес-логика ранжирования
├── models.py            # Pydantic модели для валидации
├── ranker/              # Основной пакет с ML-логикой
│   ├── const.py         # Глобальные константы и настройки
│   ├── matching_baskets.py  # Основная ML-модель для ранжирования
│   ├── modules/         # Модульные компоненты системы
│   │   ├── base.py     # Абстрактные классы и интерфейсы
│   │   ├── classifiers.py  # Вспомогательные классификаторы
│   │   └── ranking.py  # Дополнительные алгоритмы ранжирования
│   └── utils/           # Утилитарные функции
│       └── load_models.py  # Загрузка обученных моделей из файловой системы
├── requirements.txt     # Зависимости проекта
└── README.md           # Документация
```

### Компоненты системы

- **`main.py`** - HTTP API endpoints и middleware
- **`service.py`** - Сервисный слой с бизнес-логикой
- **`models.py`** - Схемы данных и валидация
- **`ranker/`** - Модули машинного обучения

## ⚙️ Конфигурация

### Параметры запроса

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `text` | string | - | Текстовый запрос (1-1000 символов) |
| `top_k` | integer | 5 | Количество возвращаемых результатов |

### Ограничения

- Максимальная длина запроса: **1000 символов**
- Максимальное количество запросов в пакете: **50**
- Максимальное значение `top_k`: зависит от данных

## 🔧 Разработка

### Структура кода

Проект следует принципам чистой архитектуры:

1. **Presentation Layer** (`main.py`) - HTTP endpoints
2. **Service Layer** (`service.py`) - бизнес-логика
3. **Data Models** (`models.py`) - схемы данных
4. **ML Layer** (`ranker/`) - модели машинного обучения

### Запуск в режиме разработки

```bash
uvicorn main:app --reload --log-level debug
```

### Логирование

Система использует стандартный модуль `logging` Python:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
