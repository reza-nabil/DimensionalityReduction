# LDAvsPCA in classification
# PCA --> best for visualization
# LDA --> best for discrimination

# import libs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


# dataset loading
iris = load_iris()


#feature extraction --> with PCA and LDA
num_components = 2

pca_obj = PCA(n_components = num_components)
X_new_pca = pca_obj.fit_transform(iris.data)

lda_obj = LDA(n_components = num_components)
X_new_lda = lda_obj.fit_transform(iris.data, iris.target)

# split new data --> to train and test Data
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(X_new_pca,
                                                                    iris.target,
                                                                    test_size = 0.3,
                                                                    random_state = 20)

x_train_lda, x_test_lda, y_train_lda, y_test_lda = train_test_split(X_new_lda,
                                                                    iris.target,
                                                                    test_size = 0.3,
                                                                    random_state = 20)


# model fiting (train model) / classification --> with KNN
num_neighbors = 5

knn_obj_pca = KNN(n_neighbors=num_neighbors)
knn_obj_pca.fit(x_train_pca, y_train_pca)


knn_obj_lda = KNN(n_neighbors=num_neighbors)
knn_obj_lda.fit(x_train_lda, y_train_lda)



# testing model (predict)
y_pred_lda = knn_obj_lda.predict(x_test_lda)
performance_lda = mse(y_test_lda, y_pred_lda)

y_pred_pca = knn_obj_pca.predict(x_test_pca)
performance_pca = mse(y_test_pca, y_pred_pca)


# show Result
print('LDA performance: ', performance_lda)
print('PCA performance: ', performance_pca)


plt.subplot(1, 2, 1)
plt.scatter(x_test_lda[:,0], x_test_lda[:,1], c=y_test_lda)
plt.title("LDA")

plt.subplot(1, 2, 2)
plt.scatter(x_test_pca[:,0], x_test_pca[:,1], c=y_test_pca)
plt.title("PCA")
