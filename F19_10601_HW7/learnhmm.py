import numpy as np
from collections import OrderedDict 
import sys


def parse_dict(dict_input):
    dictionary = OrderedDict()
    input_file = open(dict_input , 'r')
    # out = input_file.readlines()
    counter = 0
    for word in input_file:
        word = word.rstrip()
        dictionary[word]=counter
        counter+=1
    return dictionary

def parse_data(train_words_path):
    tags = []
    words = []
    input_file = open(train_words_path , 'r')
    for line in input_file:
        line_words = []
        line_tags = []
        for couple in line.split():
            temp=couple.split('_')
            line_words.append(temp[0])
            line_tags.append(temp[1])
        words.append(line_words)
        tags.append(line_tags)
    return words , tags





def learn_transition_prob(index_to_tag,tags):
    dim = len(index_to_tag)
    a = np.zeros((dim,dim))
    add_one = np.ones((dim,dim))
    for line in tags:
        if len(line) !=1:
            for i in range(len(line)-1):
                row_num = index_to_tag[line[i]]
                col_num = index_to_tag[line[i+1]]
                a[row_num,col_num] +=1
    a = a + add_one
    # print(a)
    for row in range(dim):
        row_sum = sum(a[row,:])
        for col in range(dim):
            a[row,col] = a[row,col]/row_sum
    # print(a)
    return a

def learn_emission_prob(index_to_tag,index_to_word,tags,words):
    row_dim = len(index_to_tag)
    col_dim = len(index_to_word)
    b = np.zeros((row_dim,col_dim))
    add_one = np.ones((row_dim,col_dim))
    
    for i in range(len(tags)):
        line = tags[i]
        for j in range(len(line)):
            curr_tag = line[j]
            curr_word = words[i][j]
            row_num = index_to_tag[curr_tag]
            col_num = index_to_word[curr_word]
            b[row_num,col_num] +=1
    b = b + add_one
    for row in range(row_dim):
        row_sum = sum(b[row,:])
        for col in range(col_dim):
            b[row,col] = b[row,col]/row_sum
    # print(b)
    return b 

def learn_initial_prob(tags,index_to_tag):
    row_dim = len(index_to_tag)
    pi = np.zeros(row_dim)
    add_one = np.ones(row_dim)
    for i in range(len(tags)):
        # line = tags[i]

        curr_tag = tags[i][0]
        row_num = index_to_tag[curr_tag]
        pi[row_num] +=1
    pi = pi + add_one
    row_sum = np.sum(pi)
    pi = pi/row_sum
    return pi



    
def write_2Dlist_to_file(file_path, input_list):
    out = open(file_path,'w')
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            out.write("%s " % input_list[i][j])
        out.write('\n')
    out.close()
    







def main(index_to_word_path, index_to_tag_path, train_words,hmmprior,hmmemit,hmmtrans):
    index_to_word = parse_dict(index_to_word_path)
    index_to_tag = parse_dict(index_to_tag_path)
    words , tags = parse_data(train_words_path)
    a = learn_transition_prob(index_to_tag,tags)
    b= learn_emission_prob(index_to_tag,index_to_word,tags,words)
    pi = learn_initial_prob(tags,index_to_tag)

    write_2Dlist_to_file(hmmtrans,a)
    write_2Dlist_to_file(hmmemit,b)

    hmmprior_file = open(hmmprior,'w')
    hmmprior_file.writelines("%s\n" % prob for prob in pi)
    hmmprior_file.close()
 




if __name__ == '__main__':
    # train_words_path = 'handout/fulldata/trainwords.txt'
    # index_to_word_path = 'handout/fulldata/index_to_word.txt'
    # index_to_tag_path = 'handout/fulldata/index_to_tag.txt'
    # hmmprior = 'output/hmmprior.txt'
    # hmmemit = 'output/hmmemit.txt'
    # hmmtrans = 'output/hmmtrans.txt'

    # index_to_word_path = 'handout/toydata/toy_index_to_word.txt'
    # index_to_tag_path = 'handout/toydata/toy_index_to_tag.txt'
    # train_words_path = 'handout/toydata/toytrain.txt'

    train_words_path = sys.argv[1]
    index_to_word_path = sys.argv[2]
    index_to_tag_path = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]


    main(index_to_word_path, index_to_tag_path,train_words_path,hmmprior,hmmemit,hmmtrans)
    