# Прогнозирование финансовых рынков с использованием машинного обучения

## Описание
Этот проект предназначен для прогнозирования цен на акции с использованием моделей машинного обучения. Мы используем данные о ценах акций, предварительную обработку данных, построение модели LSTM и прогнозирование будущих цен.

## Установка
1. Склонируйте репозиторий.
2. Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

## Использование
1. Сбор данных:
    ```bash
    python data_fetching.py
    ```
2. Обработка данных:
    ```bash
    python data_preprocessing.py
    ```
3. Обучение модели:
    ```bash
    python model_training.py
    ```
4. Прогнозирование:
    ```bash
    python forecasting.py
    ```
5. Оценка модели:
    ```bash
    python evaluation.py
    ```

## Лицензия
Этот проект лицензирован под лицензией MIT.