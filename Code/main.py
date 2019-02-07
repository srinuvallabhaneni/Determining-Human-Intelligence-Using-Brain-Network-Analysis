import feature_engineering as fe
import svm_api as svm
import numpy as np
import log_reg_api as log_reg
import knn_api as knn

#Function to calculate accuracy
def accuracy_score(y_test, y_pred):
    return sum(y_test == y_pred) / len(y_pred)

#Function to split data-set into training / testing
def train_test_split(X, y, test_size):
    N = len(X)
    idx = np.random.permutation(N)
    X = X[idx]
    y = y[idx].reshape(N,1)
    split = int((1 - test_size) * N)
    return X[0:split], X[split:], y[0:split], y[split:]

#Feature Engineering

basepath = 'results/'
fe.generate_feature_matrix(basepath)

feature_matrix = np.genfromtxt(basepath + 'Engineered_Features.csv', delimiter = ',')[1:,:]

print("Stage 2/2: Testing Classifiers SVM, KNN and Logistic Regression(SGD):")
test_size = 0.2
epochs = 10000
print("Stage 2/2: Number of epochs = ", epochs)
print("Stage 2/2: Train - Test ratio: "+str((1 - test_size) * 100) + "% : "
                                                                       +str(test_size * 100) + "%")
y = feature_matrix[:,-1]
X = feature_matrix[:,:-1]

#Normalizing features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

svm_accuracies = []
lr_accuracies = []
knn_accuracies = []
for i in range(epochs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

    svm_classifier = svm.SVM(kernel = 'rbf', lamda = 1)
    svm_classifier.train(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    svm_accuracies.append(accuracy_score(y_test, y_pred))

    lr_classifier = log_reg.LogisticRegression(learningRate=0.0001)
    lr_classifier.train(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)
    lr_accuracies.append(accuracy_score(y_pred, y_test))

    knn_classifier = knn.KNN(k=5)
    knn_classifier.train(X_train, X_test)
    y_pred = knn_classifier.predict(X_test, y_train)
    knn_accuracies.append(accuracy_score(y_pred, y_test))

print("Average accuracy over "+str(epochs)+" epochs :")
print("SVM    : ", str(np.mean(svm_accuracies)*100)+"%")
print("LR-SGD : ", str(np.mean(lr_accuracies)*100)+"%")
print("KNN    : ", str(np.mean(knn_accuracies)*100)+"%")
