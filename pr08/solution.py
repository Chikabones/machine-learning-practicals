import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
X = df.select_dtypes(include=[np.number]).dropna()
y = df['target'] if 'target' in df.columns else X.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_rf = time.time()
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
time_rf = time.time() - start_rf

start_gb = time.time()
gb = HistGradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
time_gb = time.time() - start_gb

train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf.predict(X_test))

train_acc_gb = accuracy_score(y_train, gb.predict(X_train))
test_acc_gb = accuracy_score(y_test, gb.predict(X_test))

comparison_table = pd.DataFrame({
    'Модель': ['Random Forest', 'Gradient Boosting'],
    'Train Accuracy': [train_acc_rf, train_acc_gb],
    'Test Accuracy': [test_acc_rf, test_acc_gb],
    'Время обучения (сек)': [time_rf, time_gb]
})

print(comparison_table)
