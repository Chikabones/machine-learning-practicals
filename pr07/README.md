# PR-7. Случайный лес

## Target
Обучить ансамблевую модель «Случайный лес» (Random Forest) и выполнить автоматический подбор оптимальных гиперпараметров через GridSearchCV.

## Details & Implementation
- Сборка конвейера с классификатором RandomForestClassifier.
- Настройка сетки гиперпараметров (
_estimators, max_depth, min_samples_leaf).
- Оценка наилучшей модели на тестовом наборе данных.