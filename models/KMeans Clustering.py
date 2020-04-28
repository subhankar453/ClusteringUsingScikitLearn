import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

data_1 = np.array([[random.randint(1,400) for i in range(2)]for j in range(50)],
                  dtype = np.float64)
data_2 = np.array([[random.randint(300,700) for i in range(2)]for j in range(50)],
                  dtype = np.float64)
data_3 = np.array([[random.randint(600,900) for i in range(2)]for j in range(50)],
                  dtype = np.float64)
data = np.append(np.append(data_1, data_2, axis = 0), data_3, axis = 0)
data.shape

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(data[:, 0], data[:, 1], s = 200)

labels_1 = np.array([0 for i in range(50)])
labels_2 = np.array([1 for i in range(50)])
labels_3 = np.array([2 for i in range(50)])
labels = np.append(np.append(labels_1, labels_2, axis = 0), labels_3, axis = 0)

df = pd.DataFrame({'data_x' : data[:, 0], 'data_y' : data[:, 1], 'labels' : labels})
colors = ['green', 'blue', 'purple']
plt.figure(figsize = (12, 8))
plt.scatter(data[:, 0], data[:, 1], c = df['labels'], s = 200,
            cmap = matplotlib.colors.ListedColormap(colors))

kmeans_model = KMeans(n_clusters = 3, max_iter = 10000).fit(data)
kmeans_model.labels_
centroids = kmeans_model.cluster_centers_
centroids

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'r', s = 250, marker = 's')
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 20)
    
print("Homegenity score : ", metrics.homogeneity_score(labels, kmeans_model.labels_))
print("Completeness score : ", metrics.completeness_score(labels, kmeans_model.labels_))
print("V_measure_score : ", metrics.v_measure_score(labels, kmeans_model.labels_))
print("Adjusted rand score : ", metrics.adjusted_rand_score(labels, kmeans_model.labels_))
print("Adjusted_mutual_info_score : ", metrics.adjusted_mutual_info_score(labels, kmeans_model.labels_))
print("Silhouette score : ", metrics.silhouette_score(data, kmeans_model.labels_))

colors = ['green', 'blue', 'purple']
plt.figure(figsize = (12, 8))
plt.scatter(data[:, 0], data[:, 1], c = df['labels'], s = 200,
            cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'r', s = 250, marker = 's')
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 30)
    
data_test = np.array([[442., 621.],
                      [50., 153.],
                      [333., 373.],
                      [835., 816.]])

label_pred = kmeans_model.predict(data_test)
label_pred

colors = ['green', 'blue', 'purple']
plt.figure(figsize = (12, 8))
plt.scatter(data[:, 0], data[:, 1], c = df['labels'], s = 200,
            cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.5)
plt.scatter(data_test[:, 0], data_test[:, 1], c = 'orange', s = 300, 
            marker = '^')
for i in range(len(data_test)):
    plt.annotate(label_pred[i], (data_test[i][0] + 7, data_test[i][1] - 7), 
                 fontsize = 30)
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'r', s = 250, marker = 's')
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 30)
