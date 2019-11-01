from __future__ import division
from collections import OrderedDict 
import numpy as np
import math
import sys




def convert_one_hot(labels,K):
    '''
    param: labels N
    return:labels K x N
    '''
    example_num=labels.shape[0]
    one_hot_labels = np.zeros((example_num,K))
    for i in range(example_num):
        one_hot_labels[i,labels[i]]=1
    
    one_hot_labels = np.transpose(one_hot_labels)
    return one_hot_labels




def initialize_params(hidden_unit_num,init_flag,feature_num, K=10):
    if init_flag ==1:
        alpha = 2*np.random.random_sample((hidden_unit_num,feature_num))-1
        beta =  2*np.random.random_sample((K,hidden_unit_num+1))-1
        
        #setting bias terms to zero
        alpha[:,0]=0
        beta [:,0]=0
        # print(alpha.shape)
        # print(alpha)
        # print(beta)
    elif init_flag ==2:
        alpha = np.zeros((hidden_unit_num,feature_num))
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




def linearForward_alpha(train_input_data,alpha):
    '''
    param: alpha: D x (m+1)
    param: train_input_data: 1 x (m+1)
    return a Dx1
    '''
    a = alpha.dot(np.transpose(train_input_data))
    return a

def sigmoid_forward(a):
    '''
    param: a D x 1
    return: z D x 1
    '''
    z = 1 / (1+np.exp(-a))
    return z
    


def linearForward_beta(z,beta):
    '''
    param: beta K x (D+1)
    param: z: D+1 x 1
    return b K x 1
    '''
    return beta.dot(z)


def softmax_forward(b):
    '''
    param: b K x 1
    return y_hat K x 1
    # '''
    # print('numerator',np.exp(b))
    # print('denominator',np.sum(np.exp(b),axis = 0))
    y_hat = np.exp(b)/np.sum(np.exp(b),axis = 0)
    return y_hat


def CrossEntropy_forward(labels,y_hat):
    '''
    param: labels: K x 1
    param: y_hat K x 1
    '''
    example_num = labels.shape[1]
    y_hat_log = np.log(y_hat)
    J = np.sum(np.multiply(labels,y_hat_log)) 
    J = -J/example_num
    return J


def NNForward(train_input_data,labels,alpha,beta):
    print('#################Forward Pass')
    print('train_input_data: %d x %d' %(train_input_data.shape[0],train_input_data.shape[1]))
    print('labels: %d x %d' %(labels.shape[0],labels.shape[1]))
    print('alpha: %d x %d' %(alpha.shape[0],alpha.shape[1]))
    # print('alpha',alpha)
    # print('beta',beta)
    a = linearForward_alpha(train_input_data,alpha)
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

    return_object = (a,z,b,y_hat,J,beta,train_input_data,labels)
    return return_object



def CrossEntropy_backward(labels, y_hat, J):
    '''
    param: labels K x 1
    param: y_hat K x 1
    param: J scalar
    return: gy_hat K x 1
    '''
    # dl/dy_hat
    gy_hat = -np.divide(labels,y_hat)
    print('gy_hat: %d x %d' %(y_hat.shape[0],y_hat.shape[1]))
    return gy_hat

def softmax_backward(labels,b,y_hat,gy_hat):
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

def linear_backward_alpha(train_input_data, a, ga):
    '''
    param: train_input_data N x m+1
    param: a D x N
    param: ga D x N 
    return: galpha D x m+1
    '''
    galpha = ga.dot(train_input_data)
    print('galpha: %d x %d' %(galpha.shape[0], galpha.shape[1]))
    return galpha

def NNBackward(forward_object):
    print('#################Backward Pass')
    a,z,b,y_hat,J,beta,train_input_data,labels = forward_object 
    gy_hat= CrossEntropy_backward(labels, y_hat, J)
    gb = softmax_backward(labels,b,y_hat,gy_hat)
    gbeta,gz = linear_backward_beta(z,b,gb,beta)
    ga = sigmoid_backward(a,z,gz)
    galpha = linear_backward_alpha(train_input_data, a, ga)

    return galpha,gbeta


def calc_error(prediction,labels):
    prediction = np.argmax(prediction,axis=0)
    # print(labels.shape)
    # print(prediction.shape)
    false_prediction_num=0
    for i in range(len(labels)):
        if labels[i]!=prediction[i]:
            false_prediction_num+=1
    error_rate = false_prediction_num/float(len(labels))
    return error_rate, prediction




def main(train_input, test_input, train_out,test_out,metrics_out,num_epoch, hidden_unit_num, init_flag, lr):
    metrics_file = open(metrics_out, 'w')
    #output class number
    K = 10
    #Proccess Training Data
    train_input_data, train_original_labels = parse_dict(train_input)
    train_labels= convert_one_hot(train_original_labels,K)   
    #append bias term
    train_input_data = np.insert(train_input_data, 0, 1,axis=1) 

    #Process Testing Data
    test_input_data, test_original_labels = parse_dict(test_input)
    test_labels= convert_one_hot(test_original_labels,K)   
    #append bias term
    test_input_data = np.insert(test_input_data, 0, 1,axis=1) 

    #initialize parameters
    feature_num = train_input_data.shape[1]
    alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)

    example_num = train_input_data.shape[0]
    for epoch in range(num_epoch):
        for example in range(example_num):
            example_data = train_input_data[example,:][np.newaxis,:] 
            example_label = train_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            # print('example_label shape',example_data.shape)
            galpha, gbeta = NNBackward(forward_object)
            alpha = alpha - lr*galpha
            beta = beta - lr*gbeta

        #generate metric file
        (a,z,b,train_prediction,J_training,beta,example_data,example_label) = NNForward(train_input_data,train_labels,alpha,beta)
        (a,z,b,test_prediction,J_testing,beta,example_data,example_label) = NNForward(test_input_data,test_labels,alpha,beta)
        metrics_file.write('epoch=%d crossentropy(train): %f\n'% (epoch+1,J_training))
        metrics_file.write('epoch=%d crossentropy(test): %f\n'% (epoch+1,J_testing))


    #make training prediction
    (a,z,b,train_prediction,J_training,beta,example_data,example_label) = NNForward(train_input_data,train_labels,alpha,beta)
    train_error_rate,train_prediction= calc_error(train_prediction,train_original_labels)

    #writing the training label file
    train_label_file = open(train_out, 'w')
    train_label_file.writelines("%s\n" % label for label in train_prediction)
    train_label_file.close()
    print('test error',train_error_rate)


    #make testing prediction
    (a,z,b,test_prediction,J_testing,beta,example_data,example_label) = NNForward(test_input_data,test_labels,alpha,beta)
    test_error_rate,test_prediction = calc_error(test_prediction,test_original_labels)

    #writing the test label file
    test_label_file = open(test_out, 'w')
    test_label_file.writelines("%s\n" % label for label in test_prediction)
    test_label_file.close()
    print('test error',test_error_rate)
            

    metrics_file.write('error(train): %f \n'% (train_error_rate))
    metrics_file.write('error(test): %f \n'% (test_error_rate))




def plotting_part1():
    output_file = 'output_files/1_2_plot.csv'
    train_input = 'handout/largeTrain.csv'
    test_input = 'handout/largeValidation.csv'
    num_epoch = 100
    lr = 0.01
    init_flag = 1
    #output class number
    K = 10
    #Proccess Training Data
    train_input_data, train_original_labels = parse_dict(train_input)
    train_labels= convert_one_hot(train_original_labels,K)   
    #append bias term
    train_input_data = np.insert(train_input_data, 0, 1,axis=1)

    #Process Testing Data
    test_input_data, test_original_labels = parse_dict(test_input)
    test_labels= convert_one_hot(test_original_labels,K)   
    #append bias term
    test_input_data = np.insert(test_input_data, 0, 1,axis=1) 

    hidden_unit_list = [5,20,50,100,200]
    train_avg_cross_entropy_list = []
    test_avg_cross_entropy_list=[]
    for hidden_unit_num in hidden_unit_list: 
        #initialize parameters
        feature_num = train_input_data.shape[1]
        alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)
        example_num = train_input_data.shape[0]

        #training loop 
        for epoch in range(num_epoch):
            for example in range(example_num):
                example_data = train_input_data[example,:][np.newaxis,:] 
                example_label = train_labels[:,example][:,np.newaxis] 
                forward_object =  NNForward(example_data,example_label,alpha,beta)
                a,z,b,y_hat,J,beta,example_data,example_label = forward_object
                # print('example_label shape',example_data.shape)
                galpha, gbeta = NNBackward(forward_object)
                alpha = alpha - lr*galpha
                beta = beta - lr*gbeta
        
        #analyze loop

        #Method 1 Training cross entropy 
        J_total=0
        for example in range(example_num):
            example_data = train_input_data[example,:][np.newaxis,:] 
            example_label = train_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            J_total += J
        train_avg_cross_entropy_list.append(J_total/example_num) 


        J_total=0
        test_example_num = test_input_data.shape[0]
        for example in range(test_example_num):
            example_data = test_input_data[example,:][np.newaxis,:] 
            example_label = test_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            J_total += J
        test_avg_cross_entropy_list.append(J_total/test_example_num) 
        
        #Metho 2 (Not sure if is correct)
        # (a,z,b,train_prediction,J,beta,example_data,example_label) = NNForward(train_input_data,train_labels,alpha,beta)
        # avg_cross_entropy_list.append(J)

    output_file = open(output_file, 'w')
    output_file.writelines('Avg train cross entropy: \n')
    output_file.writelines("%s " % J for J in train_avg_cross_entropy_list)
    output_file.writelines('\n  Avg test cross entropy: \n')
    output_file.writelines("%s " % J for J in test_avg_cross_entropy_list)
    output_file.close()

    
def potting_part2():
    output_file = 'output_files/1_3_plot_pointOOone.csv'
    train_input = 'handout/largeTrain.csv'
    test_input = 'handout/largeValidation.csv'
    num_epoch = 100
    init_flag = 1
    hidden_unit_num = 50
    lr = 0.001
    #output class number
    K = 10
    #Proccess Training Data
    train_input_data, train_original_labels = parse_dict(train_input)
    train_labels= convert_one_hot(train_original_labels,K)   
    #append bias term
    train_input_data = np.insert(train_input_data, 0, 1,axis=1)

    #Process Testing Data
    test_input_data, test_original_labels = parse_dict(test_input)
    test_labels= convert_one_hot(test_original_labels,K)   
    #append bias term
    test_input_data = np.insert(test_input_data, 0, 1,axis=1) 

    #initialize parameters
    feature_num = train_input_data.shape[1]
    alpha, beta = initialize_params(hidden_unit_num,init_flag,feature_num)

    avg_train_cross_entropy_list = []
    avg_test_cross_entropy_list = []

    train_example_num = train_input_data.shape[0]
    test_example_num = test_input_data.shape[0]
    #training loop 
    for epoch in range(num_epoch):
        for example in range(train_example_num):
            example_data = train_input_data[example,:][np.newaxis,:] 
            example_label = train_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            # print('example_label shape',example_data.shape)
            galpha, gbeta = NNBackward(forward_object)
            alpha = alpha - lr*galpha
            beta = beta - lr*gbeta

        #analyze loop
        J_total=0
        for example in range(train_example_num):
            example_data = train_input_data[example,:][np.newaxis,:] 
            example_label = train_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            J_total += J
        avg_train_cross_entropy_list.append(J_total/train_example_num) 

        J_total=0
        for example in range(test_example_num):
            example_data = test_input_data[example,:][np.newaxis,:] 
            example_label = test_labels[:,example][:,np.newaxis] 
            forward_object =  NNForward(example_data,example_label,alpha,beta)
            a,z,b,y_hat,J,beta,example_data,example_label = forward_object
            J_total += J
        avg_test_cross_entropy_list.append(J_total/test_example_num) 


    output_file = open(output_file, 'w')
    output_file.writelines('Avg train cross entropy: \n')
    output_file.writelines("%s " % J for J in avg_train_cross_entropy_list)
    output_file.writelines('\n  Avg test cross entropy: \n')
    output_file.writelines("%s " % J for J in avg_test_cross_entropy_list)
    output_file.close()





        
        






if __name__ == '__main__':
    # train_input = 'handout/largeTrain.csv'
    # test_input = 'handout/largeValidation.csv'
    # train_out = 'output_files/train_prediction.labels'
    # test_out = 'output_files/test_predictoin.labels'
    # metrics_out = 'output_files/metrics.txt'
    # num_epoch = 2
    # hidden_unit_num = 4
    # init_flag = 2
    # lr = 0.1

    # train_input = sys.argv[1]
    # test_input = sys.argv[2]
    # train_out =  sys.argv[3]
    # test_out = sys.argv[4]
    # metrics_out = sys.argv[5]
    # num_epoch = int(sys.argv[6])
    # hidden_unit_num = int(sys.argv[7])
    # init_flag = int(sys.argv[8])
    # lr = float(sys.argv[9])

    # main(train_input, test_input, train_out,test_out,metrics_out,num_epoch, hidden_unit_num, init_flag, lr)
    plotting_part1()
    # potting_part2()