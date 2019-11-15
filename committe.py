# with open('cluster_result_standardized.csv', 'r') as f1:
#     with open('lr_reclassify_result.csv', 'r') as f2:
#         with open('dt_reclassify_result.csv', 'r') as f3:
#             with open('committee.csv', 'w') as f4:
#                 line = f1.readline().strip()
#                 f4.write(line+'\r')
#                 f2.readline()
#                 f3.readline()
#                 c1 = 0
#                 c2 = 0
#                 c3 = 0
#                 c4 = 0
#                 N = 0
#                 while True:
#                     line1 = f1.readline()
#                     if not line1:
#                         break
#                     N += 1
#                     original_class = line1.strip().split(',')[0]
#                     line1 = line1.strip().split(',')[-1]
#                     line2 = f2.readline().strip().split(',')[-1]
#                     line3 = f3.readline().strip().split(',')[-1]
#                     c1 += int(line1==line2)
#                     c2 += int(line1==line3)
#                     c3 += int(line2==line3)
#                     c4 += int(line1==line2 and line2==line3)

#                     f4.write(original_class+','+str(new_class)+'\r')
#                 p1 = c1/N
#                 p2 = c2/N
#                 p3 = c3/N
#                 p4 = c4/N
#                 print("C1:{}, the proportion:{}".format(c1, p1))
#                 print("C2:{}, the proportion:{}".format(c2, p2))
#                 print("C3:{}, the proportion:{}".format(c3, p3))
#                 print("C4:{}, the proportion:{}".format(c4, p4))

with open("current_div_cost.csv", "r") as f1:
    with open("data_sum_committee_classifier_div_cost.csv", "r") as f2:
        with open("division_change.csv", "w") as f3:
            line = f1.readline().strip()
            line2 = f2.readline().strip()
            f3.write(line+",\"proportion\"\r")
            while True:
                line = f1.readline().strip()
                if not line:
                    break
                previous = float(line.strip().split(",")[-1])
                line = " ".join(line.strip().split(","))
                new = float(f2.readline().strip().split(",")[-2])
                f3.write(line+" "+"{:.2%}".format((new-previous)/previous)+"\r")
