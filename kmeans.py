from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # fix error

# Load dataset iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Clustering dengan 3 cluster
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Reduksi dimensi untuk visualisasi (PCA 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisasi hasil clustering
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title("K-Means Clustering pada Data Iris (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
