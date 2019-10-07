import sklearn
import numpy as np
import time, sys
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans  # K-means
from sklearn import metrics
from sklearn.datasets import make_blobs
from kmodes.kmodes import KModes # k-prototypes

class pipeline:

    def __init__(self):
        self.data_cleaned = 'data_cleaned.csv'
        self.cluster_result = 'cluster_result.csv'

    def data_load(self):
        temp = np.loadtxt(fname=self.data_cleaned, dtype=np.str, delimiter=',')
        self.dataset = temp[1:,3:]
        self.label = temp[0,3:]
        print(np.shape(self.dataset))
        print(self.label)
        print(self.dataset)

    def data_process(self):
        #1.process 
        

    def predict(self):
        self.y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(self.dataset)
        np.savetxt(self.cluster_result, np.hstack(self.y_pred, self.dataset) , delimiter=',')
        score = metrics.calinski_harabaz_score(self.dataset, self.y_pred)
        print(score)




pipeline = pipeline()
pipeline.data_load()
pipeline.predict()
