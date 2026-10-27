import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('=== ПРАКТИКА 1: Настройка окружения и первичный анализ данных ===')
df = sns.load_dataset('tips')

print('1. Размерность датасета:', df.shape)
print('\n2. Типы данных и общая информация:')
print(df.info())
print('\n3. Проверка пропущенных значений:\n', df.isnull().sum())
print('\n4. Описательные статистики числовых признаков:\n', df.describe())
print('\n5. Первые 5 строк датасета:\n', df.head())