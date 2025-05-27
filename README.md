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

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/62918468/dd6bb6cb-a39f-467b-b764-656f4fe20d52/tasks_for_students_mlops_25s.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/62918468/33d718d2-08a9-4823-a6c3-7fd2f004a809/project-descripton-template-mlops-pichugin.pdf
[3] https://web-ainf.aau.at/pub/jannach/files/Other_best_practices-2024.pdf
[4] https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e
[5] http://www.scitepress.org/Papers/2025/132725/132725.pdf
[6] https://mlops-coding-course.fmind.dev/6.%20Sharing/6.2.%20Readme.html
[7] https://app.readytensor.ai/publications/markdown-for-machine-learning-projects-a-comprehensive-guide-LX9cbIx7mQs9
[8] https://github.com/othneildrew/Best-README-Template
[9] https://github.com/rn5l/rsc18/blob/master/README.MD
[10] https://gitlab.com/ton1czech/project-readme-template
[11] https://git-kik.hs-ansbach.de/Ossanlou.Bijan/my-sample-project/-/blob/example-tutorial-branch/README.md
[12] https://www.youtube.com/watch?v=jeOfS90Flf8

---
Answer from Perplexity: pplx.ai/share