import numpy as np
from collections import OrderedDict 
# from learnhmm import parse_data,parse_dict
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

def write_2Dlist_to_file(file_path, input_list):
    out = open(file_path,'w')
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if j == len(input_list[i])-1:
                out.write("%s" % input_list[i][j])
            else:
                out.write("%s " % input_list[i][j])
        out.write('\n')
    out.close()



def convert_predictions_format(predictions,words):
    output = []
    for i in range(len(predictions)):
        temp_output = []
        for j in range(len(predictions[i])):
            temp_output.append(words[i][j]+'_'+predictions[i][j])
        output.append(temp_output)
    return output


def get_accuracy(predictions,tags):
    total = 0
    correct = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            total+=1
            if predictions[i][j] == tags[i][j]:
                correct +=1
    return correct/float(total)
    


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

        predictions.append(curr_prediction)
    return predictions

            
                
             


def main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans,predicted_file,metric_file):
    index_to_word = parse_dict(index_to_word_path)
    index_to_tag = parse_dict(index_to_tag_path)
    words,tags = parse_data(test_input)
    a = parse_prob_matrix(hmmtrans)
    b = parse_prob_matrix(hmmemit)
    pi =parse_prior(hmmprior)
    state_num = len(index_to_tag)
    predictions = viterbi(index_to_word,index_to_tag,words,state_num,a,b,pi)
    output = convert_predictions_format(predictions,words)
    accuracy=get_accuracy(predictions,tags)
    write_2Dlist_to_file(predicted_file,output)

    metric_file = open(metric_file,'w')
    metric_file.write('Accuracy: %0.6f' %accuracy)
    metric_file.close()




if __name__ == '__main__':
    # test_input = 'handout/fulldata/testwords.txt'
    # index_to_word_path = 'handout/fulldata/index_to_word.txt'
    # index_to_tag_path = 'handout/fulldata/index_to_tag.txt'
    # hmmprior = 'output/hmmprior.txt'
    # hmmemit = 'output/hmmemit.txt'
    # hmmtrans = 'output/hmmtrans.txt'
    # predicted_file = 'output/predicted.txt'
    # metric_file = 'output/metrics.txt'
    

    test_input = sys.argv[1]
    index_to_word_path = sys.argv[2]
    index_to_tag_path = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    # index_to_word_path = 'handout/toydata/toy_index_to_word.txt'
    # index_to_tag_path = 'handout/toydata/toy_index_to_tag.txt'
    # test_input = 'handout/toydata/toytest.txt'




    main(index_to_word_path, index_to_tag_path,test_input,hmmprior,hmmemit,hmmtrans,predicted_file,metric_file)