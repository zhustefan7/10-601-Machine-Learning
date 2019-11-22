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
    matrix = np.array(matrix)
    return matrix


def parse_prior(hmmprior):
    input_file = open(hmmprior , 'r')
    pi = []
    for line in input_file:
        pi.append(float(line))
    # print(pi)
    return pi

def get_prediction(w,p,seq_len,tag_to_index):
    # print('w',w)
    # print('p',p)
    curr_prediction = np.zeros(seq_len)
    curr_prediction[seq_len-1] = np.argmax(w[:,seq_len-1])
    # print(curr_prediction)
    for i in range(seq_len-2,-1,-1):
        # print(curr_prediction[i+1])
        # print(i)
        curr_prediction[i]=p[int(curr_prediction[i+1]),i+1]

    curr_prediction_out=[]
    for index in curr_prediction:
        curr_prediction_out.append(tag_to_index[index])

    return curr_prediction_out
        

        


def viterbi(index_to_word,index_to_tag,words,state_num,a,b,pi):
    predictions = []
    tag_to_index = {value: key for key, value in index_to_tag.items()}
    # print(words)
    for seq in words:
        # print('\n seq',seq,)
        curr_prediction = []
        w = np.zeros((state_num, len(seq)))
        p = np.zeros((state_num, len(seq)))

        for j in range(state_num):
            i = index_to_word[seq[0]]
            w[j,0] = pi[j]*b[j,i]
            p[j,0] = j
        
        
        for t in range(1,len(seq)):
            for j in range(state_num):
                i = index_to_word[seq[t]]
                w[j,t] = np.max(b[j,i]*a[:,j]*w[:,t-1])
                p[j,t] = np.argmax(b[j,i]*a[:,j]*w[:,t-1])
        
        curr_prediction = get_prediction(w,p,len(seq),tag_to_index)
        # print('curr prediction', curr_prediction)

        predictions.append(curr_prediction)
    print(predictions[2])
            
                
            


def main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans):
    index_to_word = parse_dict(index_to_word_path)
    index_to_tag = parse_dict(index_to_tag_path)
    words,tags = parse_data(test_input)
    a = parse_prob_matrix(hmmtrans)
    b = parse_prob_matrix(hmmemit)
    pi =parse_prior(hmmprior)
    # print('a',a)
    # print('b',b)
    # print('pi',pi)
    state_num = len(index_to_tag)
    viterbi(index_to_word,index_to_tag,words,state_num,a,b,pi)




if __name__ == '__main__':
    test_input = 'handout/fulldata/testwords.txt'
    index_to_word_path = 'handout/fulldata/index_to_word.txt'
    index_to_tag_path = 'handout/fulldata/index_to_tag.txt'

    # index_to_word_path = 'handout/toydata/toy_index_to_word.txt'
    # index_to_tag_path = 'handout/toydata/toy_index_to_tag.txt'
    # test_input = 'handout/toydata/toytest.txt'



    hmmprior = 'output/hmmprior.txt'
    hmmemit = 'output/hmmemit.txt'
    hmmtrans = 'output/hmmtrans.txt'

    main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans)