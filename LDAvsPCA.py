# LDAvsPCA in classification
# PCA --> best for visualization
# LDA --> best for discrimination

# import libs
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split

# dataset loading
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size = 0.3,
                                                    random_state = 20)


#feature extraction --> with PCA and LDA
num_components = 2

pca_obj = PCA(n_components = num_components)
X_new_pca = pca_obj.fit_transform(x_train)

lda_obj = LDA(n_components = num_components)
X_new_lda = lda_obj.fit_transform(x_train, y_train)


# model fiting (train model) / classification --> with KNN
num_neighbors = 5

knn_obj_pca = KNN(n_neighbors=num_neighbors)
knn_obj_pca.fit(X_new_pca, y_train)
y_pca = knn_obj_pca.predict(X_new_pca)

knn_obj_lda = KNN(n_neighbors=num_neighbors)
knn_obj_lda.fit(X_new_lda, y_train)
y_lda = knn_obj_lda.predict(X_new_lda)

# testing model (predict)
# show Result
print('LDA performance: ', np.sum((y_train == y_lda)/len(y_train)))
print('PCA performance: ', np.sum((y_train == y_pca)/len(y_train)))
