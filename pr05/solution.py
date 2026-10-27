import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

print('=== ПРАКТИКА 5: Бинарная классификация (Logistic Regression) ===')
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

clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('classifier', LogisticRegression())
])

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)
y_probs = clf_pipeline.predict_proba(X_test)[:, 1]

print('Матрица ошибок (Confusion Matrix):\n', confusion_matrix(y_test, y_pred))
print('\nОтчет о классификации (Classification Report):\n', classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}')