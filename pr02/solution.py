import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('=== ПРАКТИКА 2: Разведочный анализ данных (EDA) ===')
df = sns.load_dataset('tips')

numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

print('Корреляционная матрица Пирсона:')
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляций признаков')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print('\nТепловая карта корреляций успешно сохранена в файл correlation_matrix.png')