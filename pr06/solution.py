from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print('=== ПРАКТИКА 6: Классификация текстов (TF-IDF + Naive Bayes) ===')
corpus = [
    'great product excellent quality and service',
    'terrible waste of money bad terrible service',
    'awesome service completely satisfied experience',
    'awful quality broke immediately worst purchase',
    'highly recommend wonderful fast experience',
    'horrible do not buy garbage quality'
]
labels = [1, 0, 1, 0, 1, 0]

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(corpus)

model = MultinomialNB()
model.fit(X_vec, labels)

preds = model.predict(X_vec)
print('Размерность пространства признаков TF-IDF:', X_vec.shape)
print('Accuracy на обучающем корпусе:', accuracy_score(labels, preds))
print('\nОтчет классификации текстов:\n', classification_report(labels, preds))