import numpy as np
import pickle, json
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
np.random.seed(870963)

class pipeline:
    def __init__(self):
        self.training_set_file = 'new_training_set.csv'
        self.model_learned = 'dt_model_learned'
        self.lr_reclassify_result = 'dt_reclassify_result.csv'
        self.lr_res_stat = 'dt_res_stat_2.json'
        self.classifier = GradientBoostingRegressor()
    
    def data_load(self):
        columns = [i+3 for i in range(10)] + [i+15 for i in range(9)]
        self.original_dataset = np.loadtxt(fname=self.training_set_file, dtype=float, delimiter=',', skiprows=1, usecols=columns)
        self.training_set = np.array([row for row in self.original_dataset if row[-2]!=1])
        self.training_set_x = self.training_set[:,:-2]
        self.training_set_y = self.training_set[:,-1]#.astype(str)
        print(np.shape(self.training_set))
        print(np.shape(self.training_set_x))
        print(np.shape(self.training_set_y))

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.training_set_x, self.training_set_y, test_size=0.2)
        begin = time.time()
        print("Begin fitting...")
        self.classifier.fit(X_train, y_train)
        print("Model learned. Time used: {:.2f} s".format(time.time()-begin))
        y_pred = self.classifier.predict(X_train)
        print("Train_accuracy : %.4g" % metrics.mean_squared_error(y_train, y_pred))
        y_pred = self.classifier.predict(X_test)
        print("Test_accuracy : %.4g" % metrics.mean_squared_error(y_test, y_pred))

    def learning(self):
        begin = time.time()
        print("Begin fitting...")
        self.classifier.fit(self.training_set_x, self.training_set_y)
        print("Model learned. Time used: {:.2f} s".format(time.time()-begin))
        with open(self.model_learned, 'wb') as f:
            pickle.dump(self.classifier, f)

    def predict(self):
        with open(self.model_learned, 'rb') as f:
            self.classifier = pickle.load(f)
        self.predict_result = self.classifier.predict(self.original_dataset[:,:-2])
        original_label = self.original_dataset[:,-2:]

        f = open(self.lr_reclassify_result, 'w')
        f.write('Original class,New class\n')

        d1 = {str((x, y)):0 for x in ('Fair', 'Very Good', 'Good', 'None') for y in range(5)}
        d2 = {str((y, x)):0 for y in range(5) for x in ('Fair', 'Very Good', 'Good', 'None')}
        for row in zip(self.predict_result, original_label):
            if row[1][0] == 1:
                old = 'None'
            elif row[1][1] == 0:
                old = 'Fair'
            elif row[1][1] == 0.5:
                old = 'Good'
            elif row[1][1] == 1:
                old = 'Very Good'

            p = row[0]
            if p<0.2:
                new = 0
            elif p<0.4:
                new = 1
            elif p<0.6:
                new = 2
            elif p<0.8:
                new = 3
            else:
                new = 4

            f.write(old+','+str(new)+'\n')
            d1[str((old, new))] += 1
            d2[str((new, old))] += 1
        
        f.close()

        self.d = d1.copy()
        self.d.update(d2)
        with open(self.lr_res_stat, 'w') as f:
            json.dump(self.d, f)

        #np.savetxt(self.lr_reclassify_result, self.predict_result_prob)

    def plot(self):
        with open(self.lr_res_stat, 'r') as f:
            self.d = json.load(f)
        new_d = {i:0 for i in range(5)}
        for k, v in self.d.items():
            if str.isdigit(k[1]):
                new_d[int(k[1])] += v

        name_list = ['Poor','Fair','Good','Very Good', 'Premium']
        num_list = [new_d[0], new_d[1], new_d[2], new_d[3], new_d[4]]
        plt.bar(range(len(num_list)), num_list, color='ygcmb', tick_label=name_list)
        plt.title('Room Frequencies for Random Forest Regression')
        plt.xlabel('Space Quality')
        plt.ylabel('Number of Rooms')
        plt.show()




pipeline = pipeline()
pipeline.data_load()
pipeline.train_and_evaluate()
# pipeline.learning()
# pipeline.predict()
# pipeline.plot()
