import numpy as np
import pickle, json
import time
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
np.random.seed(870963)

class pipeline:
    def __init__(self):
        self.training_set_file = 'new_training_set.csv'
        self.model_learned = 'lr_model_learned'
        self.lr_reclassify_result = 'lr_reclassify_result.txt'
        self.lr_res_stat = 'lr_res_stat_2.json'
        self.alpha = 0.63
        self.beta = 0.02
    
    def data_load(self):
        columns = [i+3 for i in range(10)] + [i+15 for i in range(9)]
        self.original_dataset = np.loadtxt(fname=self.training_set_file, dtype=float, delimiter=',', skiprows=1, usecols=columns)
        self.training_set = np.array([row for row in self.original_dataset if row[-2]!=1])
        self.training_set_x = self.training_set[:,:-2]
        self.training_set_y = self.training_set[:,-1].astype(str)
        print(np.shape(self.training_set))
        print(np.shape(self.training_set_x))
        print(np.shape(self.training_set_y))

    def train_and_evaluate(self):
        self.classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        X_train, X_test, y_train, y_test = train_test_split(self.training_set_x, self.training_set_y, test_size=0.2)
        begin = time.time()
        print("Begin fitting...")
        self.classifier.fit(X_train, y_train)
        print("Model learned. Time used: {:.2f} s".format(time.time()-begin))
        y_pred = self.classifier.predict(X_train)
        print("Train_ccuracy : %.4g" % metrics.accuracy_score(y_train, y_pred))
        y_pred = self.classifier.predict(X_test)
        print("Test_ccuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))

    def learning(self):
        self.classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        begin = time.time()
        print("Begin fitting...")
        self.classifier.fit(self.training_set_x, self.training_set_y)
        print("Model learned. Time used: {:.2f} s".format(time.time()-begin))
        with open(self.model_learned, 'wb') as f:
            pickle.dump(self.classifier, f)

    def predict(self):
        with open(self.model_learned, 'rb') as f:
            self.classifier = pickle.load(f)
        print("classes:", self.classifier.classes_)
        self.predict_result_prob = self.classifier.predict_proba(self.original_dataset[:,:-2])

        self.predict_result = self.classifier.predict(self.original_dataset[:,:-2])
        res_stat_1 = {'0.0':0, '0.5':0, '1.0':0}
        for res in self.predict_result:
            res_stat_1[res] += 1
        print(res_stat_1)

        original_label = self.original_dataset[:,-2:]

        f = open(self.lr_reclassify_result, 'w')
        f.write('Original class\tNew class\n')

        d1 = {str((x, y)):0 for x in ('Fair', 'Very Good', 'Good', 'None') for y in range(5)}
        d2 = {str((y, x)):0 for y in range(5) for x in ('Fair', 'Very Good', 'Good', 'None')}
        for row in zip(self.predict_result_prob, original_label):
            if row[1][0] == 1:
                old = 'None'
            elif row[1][1] == 0:
                old = 'Fair'
            elif row[1][1] == 0.5:
                old = 'Good'
            elif row[1][1] == 1:
                old = 'Very Good'

            max_p = max(row[0])
            max_index = 2*list(row[0]).index(max_p)
            if max_p > self.alpha:
                new = max_index
            elif max_index == 0:
                new = 1
            elif max_index == 4:
                new = 3
            elif abs(row[0][0]-row[0][2])<self.beta:
                new = 2
            elif row[0][0]>row[0][2]:
                new = 1
            elif row[0][0]<row[0][2]:
                new = 3

            f.write(old+'\t'+str(new)+'\n')
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
        plt.show()




pipeline = pipeline()
pipeline.data_load()
# pipeline.learning()
pipeline.train_and_evaluate()
# pipeline.predict()
# pipeline.plot()
