from collections import OrderedDict 
import numpy as np
import math
import sys




def initialize_params(hidden_unit_num,init_flag,feature_num):
    K = 10
    if init_flag ==1:
        alpha = 2*np.random.random_sample((hidden_unit_num,feature_num+1))-1
        beta =  2*np.random.random_sample((K,hidden_unit_num+1))-1
    elif init_flag ==0:
        alpha = np.zeros((hidden_unit_num,feature_num+1))
        beta =  np.zeros((K,hidden_unit_num+1))
    return alpha,beta





def parse_dict(data_path):
    line_count = 0
    all_data = []
    input_file = open(data_path , 'r')
    for line in input_file:
        line = line.strip()
        data = line.split(',')
        data = map(int, data)
        all_data.append(data)
    all_data = np.array(all_data)[:,1:]
    labels = all_data[:,0]
    return all_data ,labels

def sigmoid_forward(a):
    '''
    param: a D x N
    return: z D x N
    '''
    z = 1 / (1+np.exp(-a))
    return z
    


def linearForward_alpha(input_data,alpha):
    '''
    param: alpha: D x (m+1)
    param: input_data: N x (m+1)
    return a DxN
    '''
    a = np.matmul(alpha,np.transpose(input_data))
    return a


def linearForward_beta(input_data,beta):
    '''
    param: beta K x (D+1)
    param: input_data: D+1 x N
    return b K x N
    '''
    return np.matmul(beta,input_data)


def softmax_forward(b):
    '''
    param: b K x N
    return y_hat K x N
    '''
    y_hat = np.exp(b)/np.sum(np.exp(b),axis = 0)
    return y_hat


def CrossEntropy_forward(labels,y_hat):
    '''
    param: input_data: N x (m+1)
    param: y_hat K x N
    '''
    labels = labels[:,np.newaxis]    #Nx1
    y_hat_log = np.log(y_hat)                  #KxN
    return -1/labels.shape[0]*sum(np.matmul(y_hat_log,labels))


def NNForward(input_data,lablels,alpha,beta):
    print('input_data: %d x %d' %(input_data.shape[0],input_data.shape[1]))
    #append bias term to the input_data
    input_data = np.insert(input_data, 0, 1,axis=1)
    print('input_data: %d x %d' %(input_data.shape[0],input_data.shape[1]))
    a = linearForward_alpha(input_data,alpha)
    print('a: %d x %d' %(a.shape[0],a.shape[1]))
    #activate with sigmoid
    z = sigmoid_forward(a)
    print('z: %d x %d' %(z.shape[0],z.shape[1]))

    #append bias term to z
    z = np.insert(z, 0, 1,axis=0)
    print('z: %d x %d' %(z.shape[0],z.shape[1]))

    b = linearForward_beta(z,beta)
    print('b: %d x %d' %(b.shape[0],b.shape[1]))

    y_hat = softmax_forward(b)
    print('y_hat: %d x %d' %(y_hat.shape[0],y_hat.shape[1]))

    J = CrossEntropy_forward(labels,y_hat)
    print('J', J.shape)

    return_object = (a,z,b,y_hat,J)
    return return_object







if __name__ == '__main__':


    train_input = 'handout/smallTrain.csv'
    input_data, labels = parse_dict(train_input)
    hidden_unit_num = 12
    init_flag = 1
    feature_num = input_data.shape[1]
    alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)
    NNForward(input_data,labels,alpha,beta)
    # print(all_data[:,0])
    # print(all_data) 