import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.euclidean_distance_matrix=[]

    def compute_euclidean_distance(self, X_train, test_example):
        euclidean_distance_vector = np.linalg.norm((X_train - test_example), axis=1)
        return euclidean_distance_vector

    def find_knn(self, euclidean_distance_vector):
        nearest_neighbours = np.argsort(euclidean_distance_vector)[:self.k]
        return nearest_neighbours

    def train(self, X_train, X_test):
        euclidean_distance_matrix = []
        for i in range(len(X_test)):
            self.euclidean_distance_matrix.append(self.compute_euclidean_distance(X_train, X_test[i]))


    def predict(self, X_test, y_train):
        y_train = y_train.flatten()
        output_labels = []

        for i in range(len(X_test)):
                nearest_neighbours = self.find_knn(self.euclidean_distance_matrix[i])
                neighbour_labels = []

                for neighbour in nearest_neighbours:
                        neighbour_labels.append(y_train[neighbour])

                predicted_label = np.argmax(np.bincount(neighbour_labels))
                output_labels.append(predicted_label)

        return output_labels
