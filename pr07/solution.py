import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print('=== ПРАКТИКА 7: Ансамбль «Случайный лес» (GridSearchCV) ===')
df = sns.load_dataset('tips')
df['good_tip'] = (df['tip'] / df['total_bill'] > 0.15).astype(int)

X = df.drop(columns=['tip', 'good_tip'])
y = df['good_tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, ['total_bill', 'size']),
    ('cat', cat_transformer, ['sex', 'smoker', 'day', 'time'])
])

rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, None],
    'model__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print('Лучшие гиперпараметры:', grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f'Accuracy настроенной модели на тесте: {accuracy_score(y_test, y_pred):.4f}')
print('\nОтчет классификации:\n', classification_report(y_test, y_pred))