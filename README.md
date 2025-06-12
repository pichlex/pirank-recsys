# PiRank RecSys

Рекомендательная система, основанная на алгоритме ранжирования PiRank для ранжирования документов.

## Постановка задачи

Сделать рекомендательную систему, основанную на алгоритме ранжирования PiRank

## Формат входных и выходных данных

На входе системы - набор документов + запрос. На выходе - вектор метрик релевантности

## Метрики

Ключевые метрики - NDCG, MAP, pFound.

## Валидация

Кросс-валидация с фиксированным сидом разделения

## Данные

Датасет Istella. В нем каждый набор данных состоит из пар запрос-документ, представленных в виде векторов признаков и соответствующих меток суждения о релевантности.

## Моделирование

### Бейзлайн

Сравнение с LAMBDAMART

### Основная модель

Модель Pi-Rank + маленькая быстрая языковая модель для получения эмбеддингов документов

## Внедрение

Программный пакет будет состоять из нескольких модулей, отвечающих за обучение и валидацию, а также за инференс модели (получение эмбеддингов документов не из датасета через маленькую языковую модель + инференс самого pi-rank на основе этих эмбеддингов)

## Setup

### Установка зависимостей

1. Убедитесь, что у вас установлен Python 3.9 или выше
2. Установите Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-username/pirank-recsys.git
   cd pirank-recsys
   ```

4. Установите зависимости:
   ```bash
   poetry install
   ```

   УСТАНОВИТЕ ПРЕ КОММИТ

5. Активируйте виртуальное окружение:
   ```bash
   poetry shell
   ```

### Настройка инструментов разработки

6. Установите pre-commit хуки:
   ```bash
   pre-commit install
   ```

7. Проверьте корректность настройки:
   ```bash
   pre-commit run --all-files
   ```

### Настройка DVC

8. Инициализируйте DVC (если используете удаленное хранилище):
   ```bash
   dvc remote add -d storage s3://your-bucket/path
   # или для Google Drive:
   # dvc remote add -d storage gdrive://your-folder-id
   ```

### Настройка MLflow

9. Запустите MLflow сервер локально для тестирования:
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

## Train

### Полный пайплайн обучения

Для запуска полного процесса обучения с настройками по умолчанию:

```bash
python commands.py train
```

### Обучение с кастомными параметрами

Для использования собственной конфигурации:

```bash
python commands.py train --config_name=custom_config
```

### Поэтапное выполнение

1. **Загрузка данных**:
   ```bash
   dvc pull
   ```

2. **Препроцессинг данных**:
   ```bash
   python commands.py preprocess
   ```

3. **Обучение модели**:
   ```bash
   python commands.py train_model
   ```

4. **Валидация**:
   ```bash
   python commands.py validate
   ```

### Мониторинг обучения

- Откройте MLflow UI по адресу http://127.0.0.1:8080 для отслеживания экспериментов
- Логи обучения сохраняются в директории `logs/`
- Графики метрик доступны в директории `plots/`

## Production Preparation

### Конвертация модели в ONNX

```bash
python commands.py convert_to_production --model_path=path/to/best_model.ckpt --output_dir=production_models/
```

### Конвертация в TensorRT

После создания ONNX модели:

```bash
python convert_to_production.py --model_path=path/to/best_model.ckpt --output_dir=production_models/
```

### Артефакты для продакшена

После конвертации в директории `production_models/` будут созданы:

- `model.onnx` - ONNX версия модели
- `model.trt` - TensorRT версия модели (если доступен)
- `config.yaml` - конфигурация модели
- `language_model/` - веса языковой модели

### Требования для развертывания

Минимальные зависимости для инференса:
- `torch`
- `onnxruntime` или `tensorrt`
- `transformers`
- `numpy`

## Infer

### Формат входных данных

Создайте JSON файл с запросами и документами:

```json
[
  {
    "query": "machine learning algorithms",
    "documents": [
      "Deep learning is a subset of machine learning that uses neural networks",
      "Random forests are ensemble methods that combine multiple decision trees",
      "Support vector machines are powerful classification algorithms"
    ]
  },
  {
    "query": "natural language processing",
    "documents": [
      "BERT is a transformer-based model for NLP tasks",
      "Word embeddings represent words as dense vectors",
      "Named entity recognition identifies entities in text"
    ]
  }
]
```

### Запуск инференса

```bash
python commands.py infer --model_path=path/to/model.ckpt --data_path=input.json --output_path=results.json
```

### Использование продакшен модели

Для использования ONNX модели:

```bash
python commands.py infer_onnx --model_path=production_models/model.onnx --data_path=input.json --output_path=results.json
```

### Формат выходных данных

Результат сохраняется в JSON формате:

```json
[
  {
    "query": "machine learning algorithms",
    "ranked_documents": [
      {
        "document_id": 0,
        "document": "Deep learning is a subset of machine learning that uses neural networks",
        "relevance_score": 0.95
      },
      {
        "document_id": 2,
        "document": "Support vector machines are powerful classification algorithms",
        "relevance_score": 0.87
      },
      {
        "document_id": 1,
        "document": "Random forests are ensemble methods that combine multiple decision trees",
        "relevance_score": 0.72
      }
    ]
  }
]
```

### Пример использования API

```python
from pirank_recsys.inference.predictor import PiRankPredictor

# Загрузка модели
predictor = PiRankPredictor("path/to/model.ckpt", config)

# Ранжирование документов
documents = ["Document 1", "Document 2", "Document 3"]
query = "search query"
ranked_docs = predictor.rank_documents(documents, query)
```

## Структура проекта

```
pirank-recsys/
├── configs/                 # Конфигурации Hydra
│   ├── config.yaml         # Основная конфигурация
│   ├── model/              # Конфигурации моделей
│   ├── data/               # Конфигурации данных
│   ├── training/           # Конфигурации обучения
│   └── inference/          # Конфигурации инференса
├── pirank_recsys/          # Основной пакет
│   ├── data/               # Модули работы с данными
│   ├── models/             # Реализации моделей
│   ├── training/           # Модули обучения
│   ├── inference/          # Модули инференса
│   └── utils/              # Утилиты
├── plots/                  # Графики и визуализации
├── logs/                   # Логи обучения
├── commands.py             # Основная точка входа
├── convert_to_production.py # Конвертация моделей
├── data.dvc               # DVC файл для данных
└── README.md              # Этот файл
```

## Разработка

### Запуск тестов

```bash
pytest tests/
```

### Проверка качества кода

```bash
pre-commit run --all-files
```

### Добавление новых зависимостей

```bash
poetry add package_name
```

## Лицензия

MIT License

## Автор

Александр Пичугин

