import numpy as np

class LogisticRegression:
    def __init__(self, learningRate=0.0001):
        self.max_iter = 1000
        self.learningRate = learningRate
        self.weightVector = 0

    def probability(self, X_sample):
        return 1.0 / (1.0 + np.exp(-np.dot(X_sample,self.weightVector.transpose())))

    def objective_function(self, X_train, y_train):
        val = np.matmul(X_train, self.weightVector.transpose())
        return  np.sum(np.multiply(y_train,val) - np.log(1 + np.exp(val)))


    def train(self, X_train, y_train):
        self.weightVector = np.zeros(shape=(1, X_train.shape[1]))
        threshold = 0
        cost = self.objective_function(X_train, y_train)
        cost_old = cost
        for n in range(self.max_iter):
            cost_old = cost_old + threshold
            self.learningRate /= (n+1)
            gradient = np.zeros(shape=(1,X_train.shape[1]))
            weightVectorOld = self.weightVector
            for i in range(len(y_train)):
                error = y_train[i] - self.probability(X_train[i])
                gradient += X_train[i].reshape(1,X_train.shape[1]) * error
                self.weightVector = self.weightVector + self.learningRate * gradient
            cost = self.objective_function(X_train, y_train)
            threshold = cost - cost_old
            if threshold <= 0:
                break
        self.weightVector = weightVectorOld

    def predict(self, X_test):
        return 1.0 * (self.probability(X_test) > 0.5)
