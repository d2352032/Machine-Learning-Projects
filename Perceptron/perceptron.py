from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import random


class Preceptron(object):
    def __init__(self, fileName):
      
        self.data = pd.read_csv(fileName, sep="\s+", header = None)
        self.num_of_datas = self.data.shape[0]
        self.num_of_features = self.data.shape[1] -1
        self.w = np.zeros(self.num_of_features+1)
        self.t = 0 


    def train(self, random_seed = -1, learning_speed = 1, max_iteration = 10000):

        X=self.data[self.data.columns[0:self.num_of_features]]
        y=self.data[self.num_of_features]

        if random_seed > -1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state = random_seed)
        else:
            X_train = X
            y_train = y
        #Note: if we use Preceptron model from sklean, we don't have to insert x0 = 1 lines
        #DataFrameName.insert(loc, column, value, allow_duplicates = False)
        X_train.insert(0, "x0", np.ones(len(y_train)), False)
        X_train.head()
        
        def sign_positive(num):
            if num > 0: return 1
            else: return -1

        w = np.zeros(self.num_of_features +1)
        for t in range(0, max_iteration):
            for n in range(0,len(X_train)):
                if sign_positive(np.dot(w, X_train.iloc[n])) != y_train.iloc[n]:
                    w = w + learning_speed * y_train.iloc[n] * X_train.iloc[n]
#                     print("iteration:", t, "index:", n, "mistake on:", X_train.iloc[n].name)
                    break
                elif n == len(X_train)-1:
#                     print("no mistake for iteration:",t)
                    self.w = w
                    self.t = t
                    return
        self.w = w
        self.t = t
        return

# if __name__ == '__main__':


pct = Preceptron('hw1_15_train.dat')
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