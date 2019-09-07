from decisionTree import*
import csv
import math
import numpy as np


def parse_file(file_path):
    """
    parses the tsv file and stores all the data of each attributes in a list 
    and map them to the column number of the corresponding attributes  
    """
    map = {}
    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        for i in range(len(headers)):
            map[i]=np.array([])
        for row in reader:
            for i in range(len(row)):
                map[i]=np.append(map[i],row[i])
    # print(map)
    return map


def stat_analsysis(map):
    """
    returns a dict which summarizes the data type and the count of each data 
    type for each of the attributes
    """
    data_stat = {}
    for key in map:
        data_dict = {}
        for data in map[key]:
            if data not in data_dict:
                data_dict[data]=1
            else:
                data_dict[data]+=1
        data_stat[key] = data_dict
    # print(data_stat)
    return data_stat

    

def calc_marginal_entropy(file_path):
    """
    calculates the marginal entropy of the labels of each data sheet
    """
    map = parse_label(input_file)
    majority = max(map.values())
    minority = min(map.values())
    total = float(majority + minority)
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
    return entropy

def calc_conditional_entropy(map,data_stat,attribute):
    """
    calculates the conditional 
    """
    #acquire the data info of the attribute stored in data_stat
    data_info = data_stat[attribute]
    #acquire the label info
    label_col = len(data_stat)-1
    label_info = data_stat[label_col]
    #acquire the data 
    data = map[attribute]
    labels = map[label_col]

    specific_entropy = {}
    for data_type in data_info:
        for label_type in label_info:  
            #attribute data where all data entries are equal to a speicifc value
            data_with_spec_val_idx = np.where(data==data_type)
            spec_label_idx = np.where(labels==label_type)
            intersect = np.intersect1d(data_with_spec_val_idx,spec_label_idx)
            print('#############')
            print('data type', data_type)
            print('label_type',label_type)
            print(data_with_spec_val_idx)
            print(spec_label_idx)
            print(intersect)


            # print(np.union1d(data ==data_type , labels_data =label_type))
            # data_with_spec_val = data[data==data_type]
            # #attribute data with specific value which satisfies speicifc label type
            # conditional_data = data_with_spec_val[labels==label_type]
            # temp_prob = conditional_data/data_with_spec_val
            # print(temp_prob)
            # # temp = -(len(map[attribute]==data_type and map[label_col]==label_type)/len(map[attribute]==data_type))
            # # print(temp)
        

    
    
        
            
    

if __name__ == "__main__":
    map = parse_file('handout/politicians_train.tsv')
    data_stat=stat_analsysis(map)
    calc_conditional_entropy(map,data_stat,0)

