from inspect import*
import csv
import math
import numpy as np
from collections import OrderedDict 



class DecisionTree:
    def __init__(self,left = None,right=None,decision=None,depth=0, \
        map = None, data_stat = None, attribute = None, left_route_val = None, right_route_val = None):
        self.left = left
        self.right = right
        self.decision = decision
        self.depth=depth
        self.map = map
        self.data_stat = data_stat
        self.attribute = attribute
        self.left_route_val = left_route_val
        self.right_route_val = right_route_val

def parse_file(file_path):
    """
    parses the tsv file and stores all the data of each attributes in a list 
    and map them to the column number of the corresponding attributes  
    """  
    map = OrderedDict() 
    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        for i in range(len(headers)):
            # print(headers[i])
            map[headers[i]]=np.array([])
        for row in reader:
            for i in range(len(row)):
                map[headers[i]]=np.append(map[headers[i]],row[i])
    # print(map)
    return map


def stat_analsysis(map):
    """
    returns a dict which summarizes the data type and the count of each data 
    type for each of the attributes
    """
    data_stat = OrderedDict()
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
    label_col = data_stat.keys()[-1]
    label_info = data_stat[label_col]
    label_lengths = []
    for label_poses in label_info.values():
        label_lengths.append(len(label_poses))
    # print(label_lengths)
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
    # label_col = len(data_stat)-1
    label_col = data_stat.keys()[-1]
    # print(data_stat.keys())
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
            if len(intersect_idx)==0:
                return 0
            #conditional probability of label being of specific value given speicific data value
            temp_prob = len(intersect_idx)/float(len(data_with_spec_val_idx))
            specific_entropy += temp_prob*math.log(temp_prob,2)
        specific_entropy = -specific_entropy
        prob = len(data_with_spec_val_idx)/float(len(data))
        conditional_entropy += prob * specific_entropy
    
    return conditional_entropy
            
def calc_mutual_info(map,data_stat,attribute):
    conditional_entropy = calc_conditional_entropy(map,data_stat,attribute)
    if conditional_entropy ==0:
        return 0
    marginal_entropy = calc_marginal_entropy(data_stat)
    mutual_info = marginal_entropy - conditional_entropy
    return mutual_info


def split_data(map,data_stat,attribute):
    new_maps = []
    new_data_stats=[]
    route_vals=[]
    for key in data_stat[attribute]:
        route_vals.append(key)


    #acquiring the poses of different data values of an attribute
    for data_poses in data_stat[attribute].values():
        new_map =OrderedDict()
        for key in map.keys():  
            new_map[key] = map[key][data_poses]
            new_data_stat = stat_analsysis(new_map)
        new_maps.append(new_map)
        new_data_stats.append(new_data_stat)
    return new_maps , new_data_stats, route_vals


def get_decision(data_stat, best_attribute):
    label_col = data_stat.keys()[-1]
    data_info = data_stat[label_col]
    decision = None
    best_length = 0
    for key in data_info:
        data_poses = data_info[key]
        if len(data_poses)>best_length:
            best_length = len(data_poses)
            decision = key
    return decision



def train_decision_tree(map,data_stat,depth=1,max_depth=3):
    best_attribute = 0
    max_mutual_info = 0
    all_attributes = map.keys()[0:-1]
    for attribute in all_attributes:
        curr_mutual_info = calc_mutual_info(map,data_stat,attribute)
        if curr_mutual_info >=max_mutual_info:
            max_mutual_info = curr_mutual_info
            best_attribute = attribute
    # printer(map,data_stat,depth,best_attribute)
    if len(all_attributes) == 1 or depth >=max_depth or len(data_stat[best_attribute])==1:
        decision = get_decision(data_stat, best_attribute)
        return DecisionTree(decision=decision, attribute=best_attribute, depth = depth)
    else:
        new_maps , new_data_stats ,route_vals= split_data(map,data_stat,best_attribute)
        new_maps[0].pop(best_attribute, None)
        new_maps[1].pop(best_attribute, None)
        left_data_stat = new_data_stats[0]
        right_data_stat = new_data_stats[1]
        left_route_val, right_route_val = route_vals[0], route_vals[1]
        decision = get_decision(data_stat, best_attribute)
        left = train_decision_tree(map=new_maps[0],data_stat=left_data_stat,depth=depth+1)
        right = train_decision_tree(map=new_maps[1],data_stat=right_data_stat,depth=depth+1)
        return DecisionTree(left=left , right=right, \
            attribute=best_attribute, decision=decision,depth=depth, left_route_val= left_route_val, right_route_val=right_route_val)



def printer(map,data_stat,depth,best_attribute):
    label_col = data_stat.keys()[-1] 
    if depth ==1:
        print_list1 = []
        for label in data_stat[label_col]:
            print_list1.append(len(data_stat[label_col][label]))
            print_list1.append(label)
        print('[{} {} / {} {}]').format(print_list1[0],print_list1[1],print_list1[2],print_list1[3])

    for key in data_stat[best_attribute]:
        print_list = []
        for label in data_stat[label_col]:
            intersect = np.intersect1d(data_stat[best_attribute][key], data_stat[label_col][label])
            print_list.append(len(intersect))
            print_list.append(label)
        # print(print_list)

        print(' | '*depth+str(best_attribute)+'[{} {} / {} {}]').format(print_list[0],print_list[1],print_list[2],print_list[3])



def tree_traversal(DecisionTree):
    if DecisionTree == None:
        return
    map = DecisionTree.map
    data_stat=DecisionTree.data_stat
    depth=DecisionTree.depth
    attribute = DecisionTree.attribute
    decision = DecisionTree.decision
    if DecisionTree != None:
        print('depth',depth, 'attribute',attribute,decision)
        tree_traversal(DecisionTree.left)
        tree_traversal(DecisionTree.right)



def classification_per_row(map,DecisionTree,index):
    attribute = DecisionTree.attribute
    if DecisionTree.left== None and DecisionTree.right== None:
        # print(DecisionTree.decision)
        # print(DecisionTree.depth)
        decision = DecisionTree.decision
        return decision
    else:
        if DecisionTree.left_route_val == map[attribute][index]:
            return classification_per_row(map,DecisionTree.left,index)
        else:
            return classification_per_row(map,DecisionTree.right, index)

def classification(map,DecisionTree):
    result = []
    for i in range(len(map[map.keys()[0]])):
        decision = classification_per_row(map,DecisionTree,i)
        result.append(decision)
    return result

 
def cal_error_rate(map,classification_result):
    label_col = data_stat.keys()[-1] 
    labels = map[label_col]
    matched_count = 0 
    for i in range(len(labels)):
        if labels[i] == classification_result[i]:
            matched_count+=1
    error_rate = (len(labels)-matched_count)/float(matched_count)
    print(error_rate)
    return error_rate
    # print(labels-classification_result)


def main(train_file, test_file, train_labels,test_labels,metrics_file):
    training_map = parse_file(train_file)
    testing_map = parse_file(test_file)
    data_stat=stat_analsysis(training_map)
    DecisionTree = train_decision_tree(training_map,data_stat)
    train_classification = classification(training_map, DecisionTree)
    test_classification = classification(testing_map,DecisionTree)
    train_error = cal_error_rate(training_map,train_classification)
        
if __name__ == "__main__":
    training_map = parse_file('handout/small_train.tsv')
    testing_map = parse_file('handout/small_test.tsv')
    # print(map.keys())
    data_stat=stat_analsysis(training_map)


    # new_maps , new_data_stats = split_data(map,data_stat,7)
    # mutual_info = calc_mutual_info(map, data_stat, 'Superfund_right_to_sue')
    # print(mutual_info)

    # marginal_entropy=calc_marginal_entropy(data_stat)
    # # calc_conditional_entropy(map,data_stat,0)
    # print(mutual_info)
    # print(marginal_entropy)
    DecisionTree = train_decision_tree(training_map,data_stat)
    # decision = classification_per_row(testing_map,DecisionTree,3)
    # print(decision)
    map = testing_map
    result =  classification(map,DecisionTree)
    # print(result)
    cal_error_rate(map, result)
    # print(DecisionTree.right_route_val)
    tree_traversal(DecisionTree)

