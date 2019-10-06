from __future__ import print_function
import sys
import csv
import numpy as np
from collections import OrderedDict 
import pickle



def parse_dict(dict_input):
    dictionary = {}
    input_file = open(dict_input , 'r')
    for line in input_file:
        data = line.split()
        dictionary[data[0]]=data[1]
    return dictionary

def extract_feature_aux1(input_file, output_file,dictionary):
    output_file = open(output_file , 'w')
    with open(input_file) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            words = row[1]
            label = row[0]
            words = words.split()
            output_file.write(label)
            temp_dict = {}
            for word in words:
                if word in dictionary:
                    if word not in temp_dict:
                        indx = dictionary[word]
                        output_file.write('\t'+str(indx)+':'+ '1')
                        temp_dict[word]=1
                
            output_file.write('\n')
    output_file.close()


def extract_feature_aux2(input_file, output_file,dictionary):
    output_file = open(output_file , 'w')
    with open(input_file) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            words = row[1]
            label = row[0]
            words = words.split()
            output_file.write(label)
            temp_dict = OrderedDict()
            for word in words:
                if word in dictionary:
                    if word not in temp_dict:
                        temp_dict[word]=1
                    else:
                        temp_dict[word]+=1
            for word, val in temp_dict.items():
                if val < 4:
                    indx = dictionary[word]
                    output_file.write('\t'+str(indx)+':'+ '1')
                        
            output_file.write('\n')
    output_file.close()

# def extract_feature_aux_fast1(input_file, output_file,dictionary):
#     input_file=open(input_file,'r')
#     lines = list(input_file)
#     print(lines[0])
#     # print(lines)
#     # lines=lines.split()
#     # print(lines)

#     # output_file = open(output_file , 'w')
#     # with open(input_file) as input_file:
#     #     line = input_file.readline()
#     #     while line:
#     #         print(line)



#     for row in lines:
#         words= row.split()
#         label = words[0]
#         print(label )
#         # output_file.write(label)
#         temp_dict = OrderedDict()
#         for word in words:
#             if word in dictionary:
#                 continue
#         #         if word not in temp_dict.keys():
#         #             temp_dict[word]=1
#         #         else:
#         #             temp_dict[word]+=1
#         # for word, val in temp_dict.items():
#         #     if val < 4:
#         #         indx = dictionary[word]
#                 # output_file.write('\t'+str(indx)+':'+ '1')
#         # output_file.write('\n')
#     # output_file.close()
#     return

def feature(train_input,validation_input,test_input,dict_input,formatted_train_out,formatted_validation_out,formatted_test_out,feature_flag):
    dictionary = parse_dict(dict_input)
    if feature_flag == 1:
        extract_feature_aux1(train_input, formatted_train_out,dictionary)
        extract_feature_aux1(validation_input, formatted_validation_out,dictionary)
        extract_feature_aux1(test_input, formatted_test_out,dictionary)
    if feature_flag == 2:
        extract_feature_aux2(train_input, formatted_train_out,dictionary)
        extract_feature_aux2(validation_input, formatted_validation_out,dictionary)
        extract_feature_aux2(test_input, formatted_test_out,dictionary)




if __name__ == '__main__':
    train_input = 'handout/largedata/train_data.tsv'
    test_input = 'handout/largedata/test_data.tsv'
    validation_input = 'handout/largedata/valid_data.tsv'
    formatted_train_out = 'formatted_files/formatted_train_out.tsv'
    formatted_test_out = 'formatted_files/formatted_test_out.tsv'
    formatted_validation_out = 'formatted_files/formatted_validation_out.tsv'
    feature_flag = 2
    dict_input = 'handout/dict.txt'
    feature(train_input,validation_input,test_input,dict_input,formatted_train_out,formatted_validation_out,formatted_test_out,feature_flag)
