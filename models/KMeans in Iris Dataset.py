import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

iris_df = pd.read_csv('datasets/iris.csv',
                      skiprows = 1,
                      names = ['sepal-length',
                              'sepal-width',
                              'petal-length',
                              'petal-width',
                              'class'])

iris_df = iris_df.sample(frac = 1).reset_index(drop = True)
iris_df['class'].unique()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
iris_df['class'] = label_encoder.fit_transform(iris_df['class'].astype(str))

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(iris_df['sepal-width'], iris_df['sepal-length'], s = 250)
plt.xlabel('sepal-width')
plt.ylabel('sepal-length')
plt.show()

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(iris_df['petal-width'], iris_df['petal-length'], s = 250)
plt.xlabel('petal-width')
plt.ylabel('petal-length')
plt.show()

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(iris_df['sepal-length'], iris_df['petal-length'], s = 250)
plt.xlabel('sepal-length')
plt.ylabel('petal-length')
plt.show()

iris_2D = iris_df[['sepal-length', 'petal-length']]
iris_2D = np.array(iris_2D)
kmeans_model_2D = KMeans(n_clusters = 3, max_iter = 1000).fit(iris_2D)
kmeans_model_2D.labels_
centroids_2D = kmeans_model_2D.cluster_centers_

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], c = 'r', s = 250, marker = 's')
for i in range(len(centroids_2D)):
    plt.annotate(i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize = 30)
    
iris_labels = iris_df['class']
print("Homegenity score : ", metrics.homogeneity_score(iris_labels, kmeans_model_2D.labels_))
print("Completeness score : ", metrics.completeness_score(iris_labels, kmeans_model_2D.labels_))
print("V_measure_score : ", metrics.v_measure_score(iris_labels, kmeans_model_2D.labels_))
print("Adjusted rand score : ", metrics.adjusted_rand_score(iris_labels, kmeans_model_2D.labels_))
print("Adjusted_mutual_info_score : ", metrics.adjusted_mutual_info_score(iris_labels, kmeans_model_2D.labels_))
print("Silhouette score : ", metrics.silhouette_score(iris_2D, kmeans_model_2D.labels_))

colors = ['yellow', 'blue', 'green']
plt.figure(figsize = (12,8))
plt.scatter(iris_df['sepal-length'], iris_df['petal-length'], c = iris_df['class'],
            s = 200, cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.5)
plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], c = 'r', s = 250, marker = 's')
for i in range(len(centroids_2D)):
    plt.annotate(i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize = 30)
    
iris_features = iris_df.drop('class', axis = 1)
iris_labels = iris_df['class']
kmeans_model = KMeans(n_clusters = 3).fit(iris_features)
kmeans_model.labels_
kmeans_model.cluster_centers_
print("Homegenity score : ", metrics.homogeneity_score(iris_labels, kmeans_model.labels_))
print("Completeness score : ", metrics.completeness_score(iris_labels, kmeans_model.labels_))
print("V_measure_score : ", metrics.v_measure_score(iris_labels, kmeans_model.labels_))
print("Adjusted rand score : ", metrics.adjusted_rand_score(iris_labels, kmeans_model.labels_))
print("Adjusted_mutual_info_score : ", metrics.adjusted_mutual_info_score(iris_labels, kmeans_model.labels_))
print("Silhouette score : ", metrics.silhouette_score(iris_features, kmeans_model.labels_))

