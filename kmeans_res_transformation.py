with open('cluster_result.tsv', 'r') as f1:
    with open('dt_reclassify_result.csv', 'r') as f2:
        with open('cluster_result_standardized.csv', 'w') as f3:
            line = f2.readline().strip()
            f3.write(line+'\r')
            record = {'3':0, '1':4, '2':1, '0':2, '4':3}
            while True:
                line = f1.readline()
                if not line:
                    break
                cluster = line.strip().split('\t')[-1]
                new_class = record[cluster]
                original_class = f2.readline().strip().split(',')[0]
                f3.write(original_class+','+str(new_class)+'\r')


