import sklearn
import numpy as np
import time, sys
import matplotlib.pyplot as plt
import pickle, json

from sklearn.cluster import KMeans  # K-means
from sklearn import metrics
from sklearn.datasets import make_blobs
from kmodes.kmodes import KModes 
from kmodes.kprototypes import KPrototypes

class pipeline:

    def __init__(self):
        self.data_cleaned = 'data_cleaned.csv'
        self.cluster_result = 'cluster_result.tsv'
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
                return temp_mean/temp_max
            else:
                return x/temp_max

    
        
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

        #4. Room Area as numerical value. softmax normalisation
        room_area = list(map(process_crv, self.dataset[:, 3]))
        rm_max = np.max([v for v in room_area if str(v) != 'nan'])
        room_area = np.array(list(map(lambda x:x/rm_max, room_area)))
        print("data type of room area:", type(room_area[0]))

        #5. RIV_rate as numerical value. NA as mean, softmax normalisation
        room_riv_rate = list(map(process_crv, self.dataset[:, 4]))
        temp_max = np.max([v for v in room_riv_rate if str(v) != 'nan'])
        temp_mean = np.mean([v for v in room_riv_rate if str(v) != 'nan'])
        room_riv_rate = np.array(list(map(process_crv2, room_riv_rate)))
        print("data type of RIV_rate:", type(room_riv_rate[0]))

        #6. room age as numerical value. NA as mean, softmax normalisation
        room_age = list(map(process_crv, self.dataset[:, 5]))
        temp_max = np.max([v for v in room_age if str(v) != 'nan'])
        temp_mean = np.mean([v for v in room_age if str(v) != 'nan'])
        room_age = np.array(list(map(process_crv2, room_age)))
        print("data type of room_age:", type(room_age[0]))

        #7. Assessment Condition Rating remains as float, NA as mean, softmax normalisation
        room_assessment_condition_rating = list(map(process_crv, self.dataset[:, 6]))
        temp_max = np.max([v for v in room_assessment_condition_rating if str(v) != 'nan'])
        temp_mean = np.mean([v for v in room_assessment_condition_rating if str(v) != 'nan'])
        room_assessment_condition_rating = np.array(list(map(process_crv2, room_assessment_condition_rating)))
        print("data type of Room Assessment Condition Rating:", type(room_assessment_condition_rating[0]))

        # merge all features
        self.dataset = np.column_stack((room_capacity, room_category, room_space_quality, room_area, \
            room_riv_rate, room_age, room_assessment_condition_rating))
        self.dataset[:, 0] = self.dataset[:, 0].astype(float)
        self.dataset[:, 2] = self.dataset[:, 2].astype(int)
        self.dataset[:, 3] = self.dataset[:, 3].astype(float)
        self.dataset[:, 4] = self.dataset[:, 4].astype(float)
        self.dataset[:, 5] = self.dataset[:, 5].astype(float)
        self.dataset[:, 6] = self.dataset[:, 6].astype(float)


        print(self.dataset)
        print(type(self.dataset[1][0]), type(self.dataset[1][1]), type(self.dataset[1][2]), \
            type(self.dataset[1][3]), type(self.dataset[1][4]), type(self.dataset[1][5]), type(self.dataset[1][6]))
        print(self.dataset[:, 0].dtype, self.dataset[:, 1].dtype, self.dataset[:, 2].dtype, \
            self.dataset[:, 3].dtype, self.dataset[:, 4].dtype, self.dataset[:, 5].dtype, self.dataset[:, 6].dtype)

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

        temp = np.loadtxt(fname=self.data_cleaned, dtype=object, delimiter=',')
        room_identity = temp[1:,:3]

        self.result = np.column_stack((room_identity, self.dataset, clusters))

        print(kproto.cluster_centroids_)
        # Print training statistics
        print(kproto.cost_)
        print(kproto.n_iter_)

        with open(self.result_binary, 'wb') as f:
            pickle.dump(self.result, f)
        with open('kproto_res', 'wb') as f:
            pickle.dump(kproto, f)

        with open(self.cluster_result, 'w') as f:
            re = self.result.tolist()
            for line in re:
                f.write("\t".join(list(map(str, line)))+'\n')
            
        for s, c in zip(self.label, clusters):
            print("Room identity: {}, cluster:{}".format(s, c))

    def present(self):
        with open(self.cluster_result, 'r') as f1:
            with open(self.data_cleaned, 'r') as f2:
                self.d1 = {str((x, y)):0 for x in ('Fair', 'Very Good', 'Good', 'None') for y in range(5)}
                self.d2 = {str((y, x)):0 for y in range(5) for x in ('Fair', 'Very Good', 'Good', 'None')}
                label = f2.readline()
                while True:
                    line1 = f1.readline().strip()
                    line2 = f2.readline().strip()
                    if not line1:
                        break
                    new_label = int(line1.split()[-1])
                    old_label = line2.split(',')[5]
                    self.d1[str((old_label, new_label))] += 1
                    self.d2[str((new_label, old_label))] += 1
                self.d = self.d1.copy()
                self.d.update(self.d2)
                with open('kmeans_stat.json', 'w') as f:
                    json.dump(self.d, f)

    def calculation(self):
        with open('kproto_res', 'rb') as f:
            kproto = pickle.load(f)
            matrix = kproto.cluster_centroids_
            print(matrix)
            class_0 = matrix[0][0]
            class_1 = matrix[0][1]
            class_2 = matrix[0][2]
            class_3 = matrix[0][3]
            class_4 = matrix[0][4]
            with open('calculation.txt', 'w') as f:
                for array, class_type in [(class_0, 0), (class_2, 2), (class_4, 4)]:
                    distance1 = np.sqrt(np.sum(np.square(array-class_1)))
                    distance2 = np.sqrt(np.sum(np.square(array-class_3)))
                    line = "class: {}, distance from Premium: {}, distance from Poor: {}".format(class_type, distance1, distance2)
                    print(line)
                    f.write(line+'\r')

    def plot(self):
        with open('kmeans_stat.json', 'r') as f:
            self.d = json.load(f)
        new_d = {i:0 for i in range(5)}
        for k, v in self.d.items():
            if str.isdigit(k[1]):
                new_d[int(k[1])] += v

        name_list = ['Poor','Fair','Good','Very Good', 'Premium']
        num_list = [new_d[3], new_d[2], new_d[0], new_d[4], new_d[1]]
        plt.bar(range(len(num_list)), num_list, color='ygcmb', tick_label=name_list)
        plt.show()



pipeline = pipeline()
# pipeline.data_load()
# pipeline.data_process()
# pipeline.predict()
# pipeline.present()
# pipeline.calculation()
pipeline.plot()