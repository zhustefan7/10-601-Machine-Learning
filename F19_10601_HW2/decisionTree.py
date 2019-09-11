from inspect import*
import csv
import math
import numpy as np



class DecisionTree:
    def __init__(self,left = None,right=None,decision=None,depth=0, map = None, data_stat = None, attribute = None):
        self.left = left
        self.right = right
        self.decision = decision
        self.depth=depth
        self.map = map
        self.data_stat = data_stat
        self.attribute = attribute

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
        for i in range(len(map[key])):
            data = map[key][i]
            if data not in data_dict:
                #a list which stores the position of each data point
                data_dict[data]=[i]
            else:
                data_dict[data].append(i)
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
    label_lengths = []
    for label_poses in label_info.values():
        label_lengths.append(len(label_poses))
    print(label_lengths)
    majority = max(label_lengths)
    minority = min(label_lengths)
    total = float(majority + minority)
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
    return entropy



def calc_conditional_entropy(map ,data_stat,attribute):
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


def split_data(map,data_stat,attribute):
    new_maps = []
    new_data_stats=[]
    #acquiring the poses of different data values of an attribute
    for data_poses in data_stat[attribute].values():
        new_map ={}
        for key in map.keys():  
            new_map[key] = map[key][data_poses]
            new_data_stat = stat_analsysis(new_map)
        new_maps.append(new_map)
        new_data_stats.append(new_data_stat)
        # print(len(new_map[key]))
        # print(new_map[key])
        # print('########')
        # print(new_data_stats[key])
    return new_maps , new_data_stats


def train_stump_tree(map,data_stat,depth,all_attributes):
    if map[0] == []:
        return DecisionTree()
    #find the best attributes
    #the attribute that gives the max mutual info
    best_attribute = 0
    max_mutual_info = 0
    for attribute in all_attributes:
        curr_mutual_info = calc_mutual_info(map,data_stat,attribute)
        if curr_mutual_info >=max_mutual_info:
            max_mutual_info = curr_mutual_info
            best_attribute = attribute
    return DecisionTree(attribute=best_attribute, depth=depth)


    # new_maps,new_data_stats = split_data(map,data_stat,best_attribute)


def train_decision_tree(map,data_stat,max_depth=3):
    all_attributes = map.keys()[0:-1]
    print(all_attributes)
    




    
        
if __name__ == "__main__":
    map = parse_file('handout/politicians_train.tsv')
    data_stat=stat_analsysis(map)
    new_maps , new_data_stats = split_data(map,data_stat,7)
    # mutual_info = calc_mutual_info(map, data_stat, 3)
    # marginal_entropy=calc_marginal_entropy(data_stat)
    # # calc_conditional_entropy(map,data_stat,0)
    # print(mutual_info)
    # print(marginal_entropy)
    train_decision_tree(map,data_stat)

