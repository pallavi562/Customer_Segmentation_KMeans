# customer_segmentation_kmeans.py
"""
Customer Segmentation using K-Means Clustering
RISE Internship Project 3
"""

# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
# Sample data (Replace with Mall_Customers.csv if available)
data = {
    'CustomerID': range(1, 21),
    'Age': [19, 21, 20, 23, 31, 35, 40, 49, 50, 60, 32, 33, 38, 41, 47, 52, 28, 29, 36, 48],
    'Annual Income (k$)': [15, 16, 17, 18, 25, 30, 40, 55, 60, 65, 29, 31, 42, 48, 54, 62, 21, 26, 39, 53],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 42, 60, 23, 80, 50, 38, 36, 61, 30, 82, 47, 55, 73, 35, 62]
}
df = pd.DataFrame(data)

print("\nDataset Preview:")
print(df.head())

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------------
# Step 3: Determine Optimal Clusters (Elbow Method)
# -----------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# -----------------------------
# Step 4: Apply K-Means with 4 clusters
# -----------------------------
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

print("\nCustomer Segmentation Result:")
print(df)

# -----------------------------
# Step 5: 2D Visualization (Annual Income vs Spending Score)
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 1]*scaler.scale_[1] + scaler.mean_[1],
            kmeans.cluster_centers_[:, 2]*scaler.scale_[2] + scaler.mean_[2],
            color='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('Customer Segmentation (Annual Income vs Spending Score)')
plt.legend()
plt.show()

# -----------------------------
# Step 6: 3D Visualization (Age, Income, Spending Score)
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
           c=df['Cluster'], cmap='viridis', s=80)
ax.set_title('3D Visualization of Customer Segments')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.show()
