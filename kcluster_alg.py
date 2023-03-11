# used for the determination of support and resistance levels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Path Definition
file_path = "Metatrader\data\GBPUSD_H4(noTickVolumes).csv"

# Load data from CSV file
bar_data = pd.read_csv(file_path)

# feature used in clustering
features = ["Open", "High", "Low", "Close"]

# Dropping redundant rows
bar_data = bar_data.dropna(subset=features)

# copy some of the data for clustering(can be adjusted for trading situation where you want support and resistance levels to be determined for a particular range)

used_bar_data = bar_data[features].copy()

# Scaling
# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the feature data
scaler.fit(used_bar_data)

# Transform the feature data using the scaler
scaled_feature_data = scaler.transform(used_bar_data)

# Elbow method to find the optimal number of clusters
centroids = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_feature_data)
    centroids.append(kmeans.inertia_)
plt.plot(range(1, 11), centroids)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Centroids')
plt.show()

# Fit K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(scaled_feature_data)


# Visualize the clusters
plt.scatter(scaled_feature_data[y_kmeans == 0, 0], scaled_feature_data[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(scaled_feature_data[y_kmeans == 1, 0], scaled_feature_data[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(scaled_feature_data[y_kmeans == 2, 0], scaled_feature_data[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(scaled_feature_data[y_kmeans == 3, 0], scaled_feature_data[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
