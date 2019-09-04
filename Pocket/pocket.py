from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import random


class Pocket(object):
    def __init__(self, fileName):
      
        self.data = pd.read_csv(fileName, sep="\s+", header = None)
        self.num_of_datas = self.data.shape[0]
        self.num_of_features = self.data.shape[1] -1
        self.w = np.zeros(self.num_of_features+1)
        self.w_miss = num_of_datas # number of mistake of w
        self.t = 0 
    
    def fit(self):
        return self.w

    def predict(self, fileName, w = self.w):
        data = pd.read_csv(fileName, sep="\s+", header = None)
        X=self.data[self.data.columns[0:self.num_of_features]]
        y=self.data[self.num_of_features]
        return self.num_of_mistake(w, X, y)

    def sign(num):
            if num > 0: return 1
            else: return -1
    
    def num_of_mistake(w, X_train, y_train):
            _sum = 0
            for n in range(0,len(X_train)):
                if self.sign(np.dot(w, X_train.iloc[n])) != y_train.iloc[n]:
                    _sum += 1
            return _sum


    def train(self, random_seed = -1, learning_speed = 1, max_iteration = 10000):


        

        
        # Split Data to X and Y
        X=self.data[self.data.columns[0:self.num_of_features]]
        y=self.data[self.num_of_features]


        # Get Random Sorted Data
        if random_seed > -1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state = random_seed)
        else:
            X_train = X
            y_train = y

        # Insert 0s for x0
        X_train.insert(0, "x0", np.ones(len(y_train)), False)
        X_train.head()
        


        # Start Training
        for t in range(0, max_iteration):
            for n in range(0,len(X_train)):
        
            # if sign(w'xn) != yn
                if self.sign(np.dot(self.w, X_train.iloc[n])) != y_train.iloc[n]:
                    w = self.w + learning_speed * y_train.iloc[n] * X_train.iloc[n]
            # if mistakes of w(t+1) < w(t) => store w(t+1)
                    num_of_mistake = self.num_of_mistake(w, X_train, y_train)
                    if num_of_mistake < self.w_miss:
                        self.w = w
                        self.w_miss = num_of_mistake
#                     print("iteration:", t, "index:", n, "mistake on:", X_train.iloc[n].name)
                    break
            # if no mistakes : end the loop.
                elif n == len(X_train)-1:
#                   print("no mistake for iteration:",t)
                    self.w = w
                    self.w_miss = 0
                    self.t = t
                    return
        self.t = t
        return

# if __name__ == '__main__':


pct = Pocket('hw1_15_train.dat')
tick1 = datetime.now()
_sum = 0
repeat = 20
for i in range(repeat):
    pct.train(random.randint(1,60000))
    _sum += pct.t
#     print(pct.t)
tick2 = datetime.now()
tdiff = tick2 - tick1
print("average converge iteration:",_sum / repeat, "for", repeat, "time perceptron in", tdiff.total_seconds(), "secs.")