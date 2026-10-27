import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print('=== ПРАКТИКА 4: Регрессионная модель (Linear Regression) ===')
df = sns.load_dataset('tips')
X = df.drop(columns=['tip'])
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, ['total_bill', 'size']),
    ('cat', cat_transformer, ['sex', 'smoker', 'day', 'time'])
])

model_pipeline = Pipeline([
    ('prep', preprocessor),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print(f'MAE (Средняя абсолютная ошибка): {mean_absolute_error(y_test, y_pred):.4f}')
print(f'R² (Коэффициент детерминации): {r2_score(y_test, y_pred):.4f}')