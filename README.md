# RestApiCarNum
![avatar](/img/api_avatar.jpg)
Описание задачи проекта:
Есть модель разработанная ML специалистом. 
Ее необходимо превратить в веб-сервис, 
который будет встроен в работу продуктивной системы, взаимодействие предполагается через REST API. 
В сторону сервиса будет направляться массив данных (как указано в тестовых данных), 
а в ответ сервис должен возвращать ответ из функции pick_regno. 
Сервис будет встроен в продуктивную, высоконагруженную систему (500 запросов в секунду).

### 📁 Структура проекта  
```
.
├── app/                   # Основная директория приложения.
├      ── main.py          # Точка входа FastAPI-приложения.
├      ── micromodel.cbm   # Модель от ML специалиста.
├      ── pick_regno.py    # Скрипт от ML специалиста для обращения к модели.
├      ── test_data.csv    # Небольшой объем данных для разработки и отладки сервиса.
├      ── test.py          # Скрипт для локального тестирования модели.
├── img/                   # Изображения.
├──.gitignore              # Файлы, игнорируемые Git-ом.
├── docker-compose.yaml    # Сервисы, необходимые для запуска проекта.
├── Dockerfile             # Кастомная сборка.
├── locustfile             # Скрипт для нагрузочного тестирования API.
├── README.md              # Информация о проекте.
├── requirements.txt       # Зависимости проекта.
└── task_description.txt   # Описание задачи
```

### ⚙️ Описание функционирования сервиса
Сервис работает в среде **Docker** — это обеспечивает удобство развёртывания и тестирования. Инструкция по запуску приведена ниже.  
**REST API**:
- Загружает модель **CatBoost** один раз при старте приложения.
- Принимает **POST-запрос** с JSON-данными о распознавании номерного знака.
- Обрабатывает данные через функцию `pick_regno`.
- Возвращает результат предсказания в формате JSON.

#### 🔄 Оптимизация загрузки модели
Изначально модель `CatBoost` загружалась внутри функции `pick_regno`, что приводило к высокой задержке при каждом запросе и не позволяло достичь необходимой производительности (500 RPS) при нагрузочном тестировании.

Чтобы устранить эту проблему:
- Загрузка модели была вынесена в отдельный этап — при старте приложения (`@app.on_event("startup")`).
- Это позволило **ускорить обработку запросов**.

### 🚀 Запуск проекта в Docker
~~~
docker compose up
~~~

### 📡 Доступные сервисы  
🔧 FastAPI    http://localhost:8000/	  
📘 Swagger UI http://localhost:8000/docs	  
🐍 Locust	  http://localhost:8089/  

### ✅ Проверка через test_data.csv
Заходим в контейнер restapicarnum-uvicorn-service:
~~~
docker exec -it <CONTAINER_ID> bash
~~~
Запускаем тестовый скрипт:
~~~
python test.py
~~~
Ожидаемый результат: `О041ВВ797 → {'prediction': [0.9546576119624413, 0.0453423880375588]}`

### 📈 Тестирование с помощью Locust  
Открываем веб-интерфейс:👉 http://localhost:8089/  
Указываем параметры нагрузки:  
Number of users: 800  
Spawn rate: 200  
Запускаем тест и получаем графики производительности RPS.
![RPS](/img/locust_test.png)

