import numpy as np
from collections import OrderedDict 
from learnhmm import parse_data,parse_dict
import sys



def parse_prob_matrix(matrix_path):
    input_file = open(matrix_path , 'r')
    matrix = []
    for line in input_file:
        temp = []   
        # print(line)
        for num in line.split():
            temp.append(float(num))
        matrix.append(temp)
    return matrix


    







def main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans):
    index_to_word = parse_dict(index_to_word_path)
    index_to_tag = parse_dict(index_to_tag_path)
    test_input = parse_data(test_input)
    
    a = parse_prob_matrix(hmmtrans)
    b = parse_prob_matrix(hmmemit)






if __name__ == '__main__':
    test_input = 'handout/fulldata/testwords.txt'
    index_to_word_path = 'handout/fulldata/index_to_word.txt'
    index_to_tag_path = 'handout/fulldata/index_to_tag.txt'
    hmmprior = 'output/hmmprior.txt'
    hmmemit = 'output/hmmemit.txt'
    hmmtrans = 'output/hmmtrans.txt'

    main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans)