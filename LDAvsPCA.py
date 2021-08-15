# LDAvsPCA in classification
# PCA --> best for visualization
# LDA --> best for discrimination

# import libs
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN

# dataset loading
x, y = load_iris(return_X_y=True)

#feature extraction --> with PCA and LDA
num_components = 2

pca_obj = PCA(n_components = num_components)
X_new_pca = pca_obj.fit_transform(x)

lda_obj = LDA(n_components = num_components)
X_new_lda = lda_obj.fit_transform(x,y)


# model fiting/classification --> with KNN
num_neighbors = 5
knn_obj_pca = KNN(n_neighbors=num_neighbors)
knn_obj_pca.fit(X_new_pca, y)
y_pca = knn_obj_pca.predict(X_new_pca)

knn_obj_lda = KNN(n_neighbors=num_neighbors)
knn_obj_lda.fit(X_new_lda, y)
y_lda = knn_obj_lda.predict(X_new_lda)


# show Result
print('LDA performance: ', np.sum((y == y_lda)/len(y)))
print('PCA performance: ', np.sum((y == y_pca)/len(y)))
