from __future__ import print_function
import sys
import csv
import numpy as np


def parse_dict(dict_input):
    dictionary = {}
    input_file = open(dict_input , 'r')
    for line in input_file:
        data = line.split()
        if data[0] not in dictionary.keys():
            dictionary[data[0]]=data[1]
    return dictionary

def parse_data(data):
    with open(data) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            print(row)
            print('/n')


# def feature(train_input,dict):



if __name__ == '__main__':
    train_input = 'handout/smalldata/smalltrain_data.tsv'
    dict_intput = 'handout/dict.txt'

    # parse_dict(dict_intput)
    parse_data(train_input)
