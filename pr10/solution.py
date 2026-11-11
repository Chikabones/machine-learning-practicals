import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv('data.csv')
X = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.savefig('elbow.png')
plt.close()

best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score (KMeans): {score:.4f}")

dbscan = DBSCAN(eps=0.5, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)
print(f"DBSCAN clusters found: {len(set(db_clusters)) - (1 if -1 in db_clusters else 0)}")

df['Cluster'] = clusters
df.to_csv('clustered_data.csv', index=False)
