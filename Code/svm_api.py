import numpy as np

########################################################################################################################
#                                                SVM Classifier                                                        #
########################################################################################################################

class SVM:
    def __init__(self, lamda = 1, kernel='rbf', deg = 2, gamma = -1, coef0 = 0, epsilon = 0.001):
        self.X_train = 0                        # training data matrix (Nxf)
        self.y_train = 0                        # training target vector (Nx1)
        self.lamda = lamda                      # regularization parameter. Equal to 1 by default
        self.kernel = kernel                    # kernel function options: 'rbf','linear' or 'poly'. 'rbf' is default
        self.alpha = 0                          # lagrange multiplier vector
        self.deg = deg                          # degree of polynomial for poly
        self.gamma = gamma                      # variance control. Equal to 1/no_of_features by default
        self.coef0 = coef0                      # value of constant term in poly kernel
        self.epsilon = epsilon                  # epsilon value determines maximum error permissible in training
        self.epochs =  int((lamda*epsilon)**-1) # number of iterations within which convergence is guarented
        self.label_flag = 0                     # handling case when y_train is binary (0,1) instead of (-1,1)

    #Kernel function: Input: 2 vectors/matrices x and y
    def kernelFn(self, x, y, name):
        
        #radial basis function kernel
        if name == 'rbf':
            return self.gamma * np.exp(- self.gamma * np.power(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :],
                                                                                                 axis=2), 2))
        
        #polynomial kernel
        elif name == 'poly':
            return np.power(self.coef0 + self.gamma * np.matmul(x,y.transpose()), self.deg)
        
        #Assume linear kernel if not rbf and poly
        else:
            return np.matmul(x, y.transpose())

    #Training SVM: input training data and corresponding labels(-1 or 1)
    def train(self, X_train, y_train):
        np.random.seed(0)
        #Updating class variables for future use
        N = len(y_train)
        self.alpha = np.zeros(shape=(N,1))
        self.X_train = X_train
        if np.min(y_train) == 0:
            self.y_train = y_train - 1 * (y_train == 0)
            self.label_flag = 1
        else:
            self.y_train = y_train
            
        num_feat = X_train.shape[1]
        
        if self.gamma == -1:
            self.gamma = 1 / num_feat #This is default gamma value
        
        pos_idx = np.where(self.y_train == 1)[0]
        no_of_pos_idx = len(pos_idx)
        neg_idx = np.where(self.y_train == -1)[0]
        no_of_neg_idx = len(neg_idx)
        all_idx = np.arange(N)
        
        flag=True
        for t in range(self.epochs):
            
            # Randomly pick element of alpha vector from +ve and -ve example equally
            if flag:
                i_t = neg_idx[np.random.randint(0, no_of_neg_idx)]
            else:
                i_t = pos_idx[np.random.randint(0, no_of_pos_idx)]
            flag = not flag
            
            n_t = 1 / (self.lamda * (t+1)) # learning rate during iteration t
            idx = np.delete(all_idx, i_t)  
            hinge_val = n_t * self.y_train[i_t]
            hinge_val *= sum(self.alpha[idx] * self.y_train[idx] *
                             self.kernelFn(X_train[idx], X_train[i_t].reshape(1,num_feat), self.kernel))
        
        # Check if i_t index is significant or not
            if hinge_val < 1:
                self.alpha[i_t] += 1
                
        self.alpha/=sum(self.alpha) #normalize alpha values
        np.random.seed()

    #Predict 
    def predict(self, X_test):
        prediction = np.sign(sum(self.alpha * self.y_train * self.kernelFn(self.X_train, X_test, self.kernel)))
        if self.label_flag == 1:
            prediction = (prediction == 1)
        return prediction.reshape(X_test.shape[0],1)
