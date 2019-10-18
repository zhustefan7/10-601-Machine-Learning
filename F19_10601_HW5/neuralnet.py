from __future__ import division
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
    all_data = np.array(all_data)
    labels = all_data[:,0]
    all_data = all_data[:,1:]
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
    a = alpha.dot(np.transpose(input_data))
    return a


def linearForward_beta(z,beta):
    '''
    param: beta K x (D+1)
    param: z: D+1 x N
    return b K x N
    '''
    return beta.dot(z)


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
    return -1/labels.shape[0]*sum(y_hat_log.dot(labels))


def NNForward(input_data,lablels,alpha,beta):
    print('#################Forward Pass')
    #append bias term to the input_data
    # input_data = np.insert(input_data, 0, 1,axis=1)
    print('input_data: %d x %d' %(input_data.shape[0],input_data.shape[1]))
    print('alpha: %d x %d' %(alpha.shape[0],alpha.shape[1]))
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
    print('beta: %d x %d' %(beta.shape[0],beta.shape[1]))

    y_hat = softmax_forward(b)
    print('y_hat: %d x %d' %(y_hat.shape[0],y_hat.shape[1]))

    J = CrossEntropy_forward(labels,y_hat)
    print('J', J.shape)

    return_object = (a,z,b,y_hat,J)
    return return_object



def CrossEntropy_backward(labels, y_hat, J):
    '''
    param: labels N
    param: y_hat K x N
    param: J scalar
    return: gy_hat K x N
    '''
    # dl/dy_hat
    gy_hat = -np.divide(labels,y_hat)
    print('gy_hat: %d x %d' %(y_hat.shape[0],y_hat.shape[1]))
    return gy_hat

def softmax_backward(b,y_hat,gy_hat):
    '''
    param: b K x N
    param: y_hat K x N
    param: gy_hat K x N
    return: gb K x N
    '''
    # dl/db
    gb = -(labels - y_hat)
    print('gb: %d x %d' %(gb.shape[0],gb.shape[1]))
    return gb

def linear_backward_beta(z,b,gb,beta):
    '''
    param: z D+1 x N
    param: b K x N
    param: gb K x N
    param: beta K x D+1 
    retun: gbeta K x D+1
    return gz D x N 
    '''
    # dl/dbeta
    gbeta = gb.dot(np.transpose(z))
    print('gbeta: %d x %d' %(gbeta.shape[0],gbeta.shape[1]))

    #dl/dz 
    beta_star = beta[:,1:]
    gz = np.transpose(np.transpose(gb).dot(beta_star))
    print('gz: %d x %d' %(gz.shape[0],gz.shape[1]))
    return gbeta, gz



def sigmoid_backward(a,z,gz):
    '''
    param: a D x N
    param: z K x N
    param: gz D x N
    return: ga D x N 
    '''
    dzda = np.exp(-a)/np.square(1+np.exp(-a))
    print('dzda: %d x %d' %(dzda.shape[0],dzda.shape[1]))
    ga = np.multiply(gz,dzda)
    print('ga: %d x %d' %(ga.shape[0], ga.shape[1]))
    return ga


def linear_backward_alpha(input_data, a, ga):
    '''
    param: input_data N x m+1
    param: a D x N
    param: ga D x N 
    return: galpha D x m+1
    '''
    galpha = ga.dot(input_data)
    print('galpha: %d x %d' %(galpha.shape[0], galpha.shape[1]))
    return galpha


def NNBackward(forward_object):
    a,z,b,y_hat,J = forward_object 
    gy_hat= CrossEntropy_backward(labels, y_hat, J)
    gb = softmax_backward(b,y_hat,gy_hat) 
    gbeta,gz = linear_backward_beta(z,b,gb,beta)
    ga = sigmoid_backward(a,z,gz)
    galpha = linear_backward_alpha(input_data, a, ga)





# def main(train_input, num_epoch, hidden_unit_num, init_flag, learning_rate):
#     input_data, labels = parse_dict(train_input)
#     feature_num = input_data.shape[1]
#     alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)





if __name__ == '__main__':
    train_input = 'handout/smallTrain.csv'
    input_data, labels = parse_dict(train_input)
    hidden_unit_num = 12
    init_flag = 1
    feature_num = input_data.shape[1]
    alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)
    input_data = np.insert(input_data, 0, 1,axis=1)
    a,z,b,y_hat,J = NNForward(input_data,labels,alpha,beta)

    gy_hat= CrossEntropy_backward(labels, y_hat, J)
    gb = softmax_backward(b,y_hat,gy_hat) 
    gbeta,gz = linear_backward_beta(z,b,gb,beta)
    ga = sigmoid_backward(a,z,gz)
    galpha = linear_backward_alpha(input_data, a, ga)
    # print(labels[1] )
    # print(y_hat[2,1])
    # # print(gy_hat[2,1])
    # print(gb[2,1])
    # print(all_data[:,0])
    # print(all_data) 