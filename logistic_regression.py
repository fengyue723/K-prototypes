import numpy as np
import pickle
import time


from sklearn.linear_model.logistic import LogisticRegression


class pipeline:
    def __init__(self):
        self.training_set_file = 'new_training_set.csv'
        self.model_learned = 'lr_model_learned'
        self.lr_reclassify_result = 'lr_reclassify_result.txt'
    
    def data_load(self):
        columns = [i+3 for i in range(10)] + [i+15 for i in range(9)]
        self.original_dataset = np.loadtxt(fname=self.training_set_file, dtype=float, delimiter=',', skiprows=1, usecols=columns)
        self.training_set = np.array([row for row in self.original_dataset if row[-2]!=1])
        self.training_set_x = self.training_set[:,:-2]
        self.training_set_y = self.training_set[:,-1].astype(str)
        print(np.shape(self.training_set))
        print(np.shape(self.training_set_x))
        print(np.shape(self.training_set_y))

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
        
        np.savetxt(self.lr_reclassify_result, self.predict_result_prob)




pipeline = pipeline()
pipeline.data_load()
#pipeline.learning()
pipeline.predict()
