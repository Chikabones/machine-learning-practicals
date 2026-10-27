import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print('=== ПРАКТИКА 3: Пайплайн предобработки данных ===')
df = sns.load_dataset('tips')

X = df.drop(columns=['tip'])
y = df['tip']

# Защита от утечки данных: разделение до трансформаций
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['total_bill', 'size']
cat_features = ['sex', 'smoker', 'day', 'time']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

print('Размерность тренировочной матрицы после трансформации:', X_train_prepared.shape)
print('Размерность тестовой матрицы после трансформации:', X_test_prepared.shape)