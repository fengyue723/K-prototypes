import sklearn
import numpy as np
import time, sys
import matplotlib.pyplot as plt
import pickle, json

from sklearn.cluster import KMeans  # K-means
from sklearn import metrics, feature_selection
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
        #process 'NA' value by indicator (return (1, 0) when NA, otherwise (0, true value))
        def process_numerical_with_indicator(x): 
            if x == 'NA':
                return (1, 0)
            else:
                return (0, float(x))

        def process_quality(x):
            if x == 'Very Good':
                return (0, 1)
            elif x == 'Good':
                return (0, 0.5)
            elif x == 'Fair':
                return (0, 0)
            else:
                return (1, 0)

        def process_category(x):
            res = [0 for _ in range(9)]
            if x != 'NA':
                res[int(x)-1] = 1
            else:
                res[-1] = 1
            return res

    
        
        #1. room_capacity: NA as a new feature, softmax normalisation
        room_capacity_indicator, room_capacity_value = zip(*list(map(process_numerical_with_indicator, self.dataset[:, 0])))
        capacity_max = np.max(room_capacity_value)
        room_capacity_indicator = np.array(room_capacity_indicator)
        room_capacity_value = np.array(list(map(lambda x:x/capacity_max, room_capacity_value)))
        print("data type of Room capacity:", type(room_capacity_value[0]))

        #2. Room Category remain as str variable
        room_category_1, room_category_2, room_category_3, room_category_4, room_category_5, room_category_6, \
            room_category_7, room_category_8, room_category_NA = zip(*list(map(process_category, self.dataset[:, 1])))
        print("data type of Room Category:", type(room_category_1[0]))

        #3. Space Quality as ordinal variable. 'Very Good' as 1, 'Good' as 0.5, 'Fair' as 0
        room_space_quality_indicator, room_space_quality_value = zip(*list(map(process_quality, self.dataset[:, 2])))
        room_space_quality_indicator = np.array(room_space_quality_indicator)
        room_space_quality_value = np.array(room_space_quality_value)
        print("data type of space quality:", type(room_space_quality_value[0]))

        #4. Room Area as numerical value. softmax normalisation
        room_area_indicator, room_area_value = zip(*list(map(process_numerical_with_indicator, self.dataset[:, 3])))
        temp_max = np.max(room_area_value)
        room_area_indicator = np.array(room_area_indicator)
        room_area_value = np.array(list(map(lambda x:x/temp_max, room_area_value)))
        print("data type of room area:", type(room_area_value[0]))

        #5. RIV_rate as numerical value. softmax normalisation
        room_riv_rate_indicator, room_riv_rate_value = zip(*list(map(process_numerical_with_indicator, self.dataset[:, 4])))
        temp_max = np.max(room_riv_rate_value)
        room_riv_rate_indicator = np.array(room_riv_rate_indicator)
        room_riv_rate_value = np.array(list(map(lambda x:x/temp_max, room_riv_rate_value)))
        print("data type of RIV_rate:", type(room_riv_rate_value[0]))

        #6. room age as numerical value. NA as mean, softmax normalisation
        room_age_indicator, room_age_value = zip(*list(map(process_numerical_with_indicator, self.dataset[:, 5])))
        temp_max = np.max(room_age_value)
        room_age_indicator = np.array(room_age_indicator)
        room_age_value = np.array(list(map(lambda x:x/temp_max, room_age_value)))
        print("data type of room_age:", type(room_age_value[0]))

        #7. Assessment Condition Rating remains as float, NA as mean, softmax normalisation
        room_assessment_condition_rating_indicator, room_assessment_condition_rating_value = zip(*list(map(process_numerical_with_indicator, self.dataset[:, 5])))
        temp_max = np.max(room_assessment_condition_rating_value)
        room_assessment_condition_rating_indicator = np.array(room_assessment_condition_rating_indicator)
        room_assessment_condition_rating_value = np.array(list(map(lambda x:x/temp_max, room_assessment_condition_rating_value)))
        print("data type of Room Assessment Condition Rating:", type(room_assessment_condition_rating_value[0]))

        # merge all features
        self.dataset = np.column_stack((room_capacity_indicator, room_capacity_value, room_category_1,\
             room_category_2, room_category_3, room_category_4, room_category_5, room_category_6, \
            room_category_7, room_category_8, room_category_NA,\
            room_area_indicator, room_area_value, room_riv_rate_indicator, room_riv_rate_value,\
            room_age_indicator, room_age_value, room_assessment_condition_rating_indicator, \
                room_assessment_condition_rating_value, room_space_quality_indicator, room_space_quality_value))
        self.dataset[:] = self.dataset[:].astype(float)

        temp = np.loadtxt(fname=self.data_cleaned, dtype=object, delimiter=',')
        room_identity = temp[1:,:3]

        self.new_training_set = np.column_stack((room_identity, self.dataset))
        title = ['No.', 'Building Code','Room Code', 'room_capacity_indicator', 'room_capacity_value',\
            'room_category_1', 'room_category_2', 'room_category_3', 'room_category_4', 'room_category_5', \
            'room_category_6', 'room_category_7', 'room_category_8', 'room_category_NA', \
            'room_area_indicator', 'room_area_value', 'room_riv_rate_indicator', 'room_riv_rate_value',\
            'room_age_indicator', 'room_age_value', 'room_assessment_condition_rating_indicator', \
            'room_assessment_condition_rating_value', 'room_space_quality_indicator', 'room_space_quality_value']

        self.new_training_set = np.row_stack((title, self.new_training_set))

        #np.savetxt('new_training_set.csv', self.new_training_set, delimiter=',')

        with open('new_training_set.csv', 'w') as f:
            re = self.new_training_set.tolist()
            for line in re:
                f.write(",".join(list(map(str, line)))+'\n')

        # with open(self.data_processed, 'wb') as f:
        #     pickle.dump(self.dataset, f)

        # with open(self.label_file, 'wb') as f:
        #     pickle.dump(self.label, f)


    def feature_selection(self):
        columns = [i+3 for i in range(10)] + [i+15 for i in range(9)]
        self.original_dataset = np.loadtxt(fname='new_training_set.csv', dtype=float, delimiter=',', skiprows=1, usecols=columns)
        self.training_set = np.array([row for row in self.original_dataset if row[-2]!=1])
        self.training_set_x = self.training_set[:,:-2]
        self.training_set_y = self.training_set[:,-1].astype(str)
        print(feature_selection.chi2(self.training_set_x, self.training_set_y))
        print(feature_selection.f_classif(self.training_set_x, self.training_set_y))



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
                with open('temp.json', 'w') as f:
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
        with open('temp.json', 'r') as f:
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
pipeline.data_load()
# pipeline.data_process()
pipeline.feature_selection()
# pipeline.predict()
# pipeline.present()
# pipeline.calculation()
# pipeline.plot()
