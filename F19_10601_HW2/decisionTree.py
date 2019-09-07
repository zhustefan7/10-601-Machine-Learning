from inspect import*
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

    

# def calc_marginal_entropy(file_path):
#     """
#     calculates the marginal entropy of the labels of each data sheet
#     """
#     map = parse_label(file_path)
#     majority = max(map.values())
#     minority = min(map.values())
#     total = float(majority + minority)
#     entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
#     return entropy

def calc_marginal_entropy(data_stat):
    """
    calculates the marginal entropy of the labels of each data sheet
    """
    label_col = len(data_stat)-1
    label_info = data_stat[label_col]
    majority = max(label_info.values())
    minority = min(label_info.values())
    total = float(majority + minority)
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
    return entropy



def calc_conditional_entropy(map,data_stat,attribute):
    """
    calculates the conditional entropy of a specific attribute
    """
    #acquire the data info of the attribute stored in data_stat
    data_info = data_stat[attribute]
    #acquire the label info
    label_col = len(data_stat)-1
    label_info = data_stat[label_col]
    #acquire the data 
    data = map[attribute]
    labels = map[label_col]
    conditional_entropy =0
    for data_type in data_info:
        specific_entropy = 0
        for label_type in label_info:  
            #attribute data indices where all data entries are equal to a speicifc value
            data_with_spec_val_idx = np.where(data==data_type)[0]
            #label indices where all labels are of same value
            spec_label_idx = np.where(labels==label_type)
            #the intersection of the two indices above
            intersect_idx = np.intersect1d(data_with_spec_val_idx,spec_label_idx)
            #conditional probability of label being of specific value given speicific data value
            temp_prob = len(intersect_idx)/float(len(data_with_spec_val_idx))
            specific_entropy += temp_prob*math.log(temp_prob,2)
        specific_entropy = -specific_entropy
        prob = len(data_with_spec_val_idx)/float(len(data))
        conditional_entropy += prob * specific_entropy
    
    return conditional_entropy
            
def calc_mutual_info(map,data_stat,attribute):
    conditional_entropy = calc_conditional_entropy(map,data_stat,attribute)
    marginal_entropy = calc_marginal_entropy(data_stat)
    mutual_info = marginal_entropy - conditional_entropy
    return mutual_info

    
        
if __name__ == "__main__":
    map = parse_file('handout/politicians_train.tsv')
    data_stat=stat_analsysis(map)
    mutual_info = calc_mutual_info(map, data_stat, 3)
    # marginal_entropy=calc_marginal_entropy2(data_stat)
    # calc_conditional_entropy(map,data_stat,0)
    print(mutual_info)

