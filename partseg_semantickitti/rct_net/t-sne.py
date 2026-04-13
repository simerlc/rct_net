import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. 生成高维数据
n_samples = 500
n_features = 50
n_clusters = 3
X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# 2. 使用t-SNE进行降维到2D
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# 3. 使用KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X_embedded)

# 4. 绘制聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.title("t-SNE and KMeans Clustering")
plt.show()
