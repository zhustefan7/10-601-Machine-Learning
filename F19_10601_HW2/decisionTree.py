from decisionTree import*
import csv
import math


def parse_file(file_path):
    map = {}
    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        for i in range(len(headers)):
            map[i]=[]
        for row in reader:
            for i in range(len(row)):
                map[i].append(row[i])
    return map


            


def calc_marginal_entropy(file_path):
    map = parse_label(input_file)
    majority = max(map.values())
    minority = min(map.values())
    total = float(majority + minority)
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
    return entropy

def calc_conditional_entropy(map,attribute):
    data_type = {}
    for data in map[attribute]:
        if data not in data_type:
            data_type[data] = 1 
        else: 
            data_type[data] +=1
            
    








if __name__ == "__main__":
    parse_file('handout/politicians_train.tsv'  )

