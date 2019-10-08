import sklearn
import numpy as np
import time, sys
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans  # K-means
from sklearn import metrics
from sklearn.datasets import make_blobs
from kmodes.kmodes import KModes 
from kmodes.kprototypes import KPrototypes

class pipeline:

    def __init__(self):
        self.data_cleaned = 'data_cleaned.csv'
        self.cluster_result = 'cluster_result.csv'
        self.data_processed = 'data_processed_file'
        self.label_file = 'label_file'
        self.result_binary = 'result_binary_file'

    def data_load(self):
        temp = np.loadtxt(fname=self.data_cleaned, dtype=object, delimiter=',')
        self.dataset = temp[1:,3:]
        self.label = temp[0,3:].astype(str)
        print(np.shape(self.dataset))
        print(self.label)
        print(self.dataset)

    def data_process(self):
        #process 'NA' value and normalisation
        def process_capacity(x):
            if x == 'NA':
                return 0
            else:
                return int(x)

        def process_quality(x):
            if x == 'Very Good':
                return 3
            elif x == 'Good':
                return 2
            elif x == 'Fair':
                return 1
            else:
                return 2

        def process_crv(x):
            if x == 'NA':
                return np.nan
            else:
                return float(x)

        def process_crv2(x):
            if str(x) == 'nan':
                return crv_mean/crv_max
            else:
                return x/crv_max

    
        
        #1. room_capacity: NA as 0, softmax normalisation
        room_capacity = list(map(process_capacity, self.dataset[:, 0]))
        capacity_max = np.max(room_capacity)
        room_capacity = np.array(list(map(lambda x:x/capacity_max, room_capacity)))
        print("data type of Room capacity:", type(room_capacity[0]))

        #2. Room Category remain as str variable
        room_category = self.dataset[:, 1]
        print("data type of Room Category:", type(room_category[0]))

        #3. Space Quality as ordinal variable. 'Very Good' as 3, 'Good' as 2, 'Fair' as 1
        room_space_quality = np.array(list(map(process_quality, self.dataset[:, 2])))
        print("data type of space quality:", type(room_space_quality[0]))

        #4. CRV_rate as numerical value. NA as mean, softmax normalisation
        room_crv_rate = list(map(process_crv, self.dataset[:, 3]))
        crv_max = np.max([v for v in room_crv_rate if str(v) != 'nan'])
        crv_mean = np.mean([v for v in room_crv_rate if str(v) != 'nan'])
        room_crv_rate = np.array(list(map(process_crv2, room_crv_rate)))
        print("data type of Room capacity:", type(room_crv_rate[0]))

        #5. Assessment Condition Rating remains as float, NA as mean, softmax normalisation
        room_assessment_condition_rating = list(map(process_crv, self.dataset[:, 4]))
        crv_max = np.max([v for v in room_assessment_condition_rating if str(v) != 'nan'])
        crv_mean = np.mean([v for v in room_assessment_condition_rating if str(v) != 'nan'])
        room_assessment_condition_rating = np.array(list(map(process_crv2, room_assessment_condition_rating)))
        print("data type of Room Assessment Condition Rating:", type(room_assessment_condition_rating[0]))

        # merge all features
        self.dataset = np.column_stack((room_capacity, room_category, room_space_quality, room_crv_rate, room_assessment_condition_rating))
        self.dataset[:, 0] = self.dataset[:, 0].astype(float)
        self.dataset[:, 2] = self.dataset[:, 2].astype(int)
        self.dataset[:, 3] = self.dataset[:, 3].astype(float)
        self.dataset[:, 4] = self.dataset[:, 4].astype(float)


        print(self.dataset)
        print(type(self.dataset[1][0]), type(self.dataset[1][1]), type(self.dataset[1][2]), type(self.dataset[1][3]), type(self.dataset[1][4]))
        print(self.dataset[:, 0].dtype, self.dataset[:, 1].dtype, self.dataset[:, 2].dtype, self.dataset[:, 3].dtype, self.dataset[:, 4].dtype)

        with open(self.data_processed, 'wb') as f:
            pickle.dump(self.dataset, f)

        with open(self.label_file, 'wb') as f:
            pickle.dump(self.label, f)
        

    def predict(self):
        with open(self.data_processed, 'rb') as f:
            self.dataset = pickle.load(f)
        with open(self.label_file, 'rb') as f:
            self.label = pickle.load(f)
        
        # self.y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(self.dataset)
        # np.savetxt(self.cluster_result, np.hstack(self.y_pred, self.dataset) , delimiter=',')
        # score = metrics.calinski_harabaz_score(self.dataset, self.y_pred)
        # print(score)
        kproto = KPrototypes(n_clusters=5, init='Cao', verbose=2)
        clusters = kproto.fit_predict(self.dataset, categorical=[1])
        self.result = np.column_stack((self.dataset, clusters))

        print(kproto.cluster_centroids_)
        # Print training statistics
        print(kproto.cost_)
        print(kproto.n_iter_)

        with open(self.result_binary, 'wb') as f:
            pickle.dump(self.result, f)

        with open(self.cluster_result, 'w') as f:
            re = self.result.tolist()
            for line in re:
                f.write(",".join(list(map(str, line)))+'\n')
            
        for s, c in zip(self.label, clusters):
            print("Room identity: {}, cluster:{}".format(s, c))




pipeline = pipeline()
# pipeline.data_load()
# pipeline.data_process()
pipeline.predict()
