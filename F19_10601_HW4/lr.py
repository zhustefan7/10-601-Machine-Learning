from feature import parse_dict
from collections import OrderedDict 
import numpy as np
import math
import sys



def create_data_dict(formatted_data, theta_origin_len):
    """this function reads in the tsv file and convert the tsv into 
   a dictionary,with each key corresponding to an example   
   and the val corresponds to indices. Also returns a labels list 
   with all the labels in order
   returns: dict, labels
   """
    formatted_data = open(formatted_data  ,'r')
    data_dict = OrderedDict()
    #to create a list of indices for each example and store them in data_dict
    example_num = 0 
    labels =[]
    for example in formatted_data:
        #stores all the indices of ones in an example
        example_indices = []
        example = example.split()
        labels.append(int(example[0]))
        for entry in example[1:]:
            entry = entry.split(':')
            example_indices.append(int(entry[0]))
            
        #appending 1 to the back of the data to account for bias term
        example_indices.append(theta_origin_len)
        data_dict[example_num] = example_indices
        example_num +=1 
    return data_dict, labels


def sparse_dot(example_indices,theta):
    return sum(theta[example_indices])


def lr_aux(data_dict, labels, dictionary, epoch ,rate):
    #initialized theta 
    theta = np.zeros(len(dictionary)+1)
    #loop through training epoch
    for i in range(epoch):
        #loop through all the data point
        for key in data_dict:
            label=labels[key]
            curr_example_indices = data_dict[key]
            product = sparse_dot(curr_example_indices ,theta)
            grad = math.exp(product)/(1+math.exp(product))-label
            theta[curr_example_indices] -= rate*grad
    return theta

def lr_aux2(data_dict, labels, dictionary, epoch ,rate,neg_log_out):
    #initialized theta 
    theta = np.zeros(len(dictionary)+1)
    neg_log_likihood=0
    neg_log_out = open(neg_log_out, 'w')
    #loop through training epoch
    for i in range(epoch):
        neg_log_likihood=0
        #loop through all the data point
        for key in data_dict:
            label=labels[key]
            curr_example_indices = data_dict[key]
            product = sparse_dot(curr_example_indices ,theta)
            grad = math.exp(product)/(1+math.exp(product))-label
            theta[curr_example_indices] -= rate*grad
            neg_log_likihood += -label*product+math.log(1+math.exp(product))
        result = neg_log_likihood/len(data_dict.keys())
        neg_log_out.write("%s\n" % str(result))
    neg_log_out.close()
    
    return theta


def prediction(data_dict, theta):
    predictions =[]
    for key in data_dict:
        curr_example_indices = data_dict[key]
        product = sparse_dot(curr_example_indices ,theta)
        raw_prediction = math.exp(product)/(1+math.exp(product))
        if raw_prediction<0.5:
            predictions.append(0)
        elif raw_prediction>=0.5:
            predictions.append(1)
    return predictions
  


def calc_error_rate(labels,predictions):
    false_prediction_num=0
    for i in range(len(labels)):
        if labels[i]!=predictions[i]:
            false_prediction_num+=1
    return false_prediction_num/float(len(labels))


def main(formatted_train,formatted_valid,formatted_test,dict_input,train_out, test_out,metrics_out,epoch):
    rate = 0.1
    dictionary = parse_dict(dict_input)
    theta_origin_len = len(dictionary)

    #training , predicting and calculating training error
    train_data_dict, train_labels = create_data_dict(formatted_train,theta_origin_len)
    theta =lr_aux(train_data_dict, train_labels, dictionary, epoch ,rate)
    train_predictions = prediction(train_data_dict,theta)
    train_err_rate = calc_error_rate(train_labels, train_predictions)

    #predict on testing data and calculate testing error
    test_data_dict, test_labels = create_data_dict(formatted_test,theta_origin_len)
    test_predictions = prediction(test_data_dict,theta)
    test_err_rate = calc_error_rate(test_labels, test_predictions)


    #writing the training label file
    train_label_file = open(train_out, 'w')
    train_label_file.writelines("%s\n" % label for label in train_predictions)
    train_label_file.close()

    #writing the testing label file
    test_label_file = open(test_out, 'w')
    test_label_file.writelines("%s\n" % label for label in test_predictions)
    test_label_file.close()

    #Writing the metric file
    metrics_file = open(metrics_out, 'w')
    metrics_file.write('error(train): %.6f\n'% train_err_rate)
    metrics_file.write('error(test): %.6f\n'% test_err_rate)
    metrics_file.close()


    print 'train_err_rate',train_err_rate
    print 'test_err_rate',test_err_rate


    

def average_log_liklihoold():
    formatted_train = 'formatted_files/formatted_train_out.tsv'
    formatted_valid =  'formatted_files/formatted_validation_out.tsv'
    dict_input = 'handout/dict.txt'

    epoch = 200
    rate = 0.1
    dictionary = parse_dict(dict_input)
    theta_origin_len = len(dictionary)

    train_neg_likihood_out = 'formatted_files/train_neg_likihood_model2.tsv'
    train_data_dict, train_labels = create_data_dict(formatted_train,theta_origin_len)
    theta =lr_aux2(train_data_dict, train_labels, dictionary, epoch ,rate,train_neg_likihood_out)

    valid_neg_likihood_out = 'formatted_files/valid_neg_likihood_model2.tsv'
    valid_data_dict, valid_labels = create_data_dict(formatted_valid,theta_origin_len)
    theta =lr_aux2(valid_data_dict, valid_labels, dictionary, epoch ,rate,valid_neg_likihood_out)

    


def avg_log_likelihood2():
    formatted_train = 'formatted_files/formatted_train_out.tsv'
    formatted_valid =  'formatted_files/formatted_validation_out.tsv'
    dict_input = 'handout/dict.txt'
    rate = 0.1
    dictionary = parse_dict(dict_input)
    theta_origin_len = len(dictionary)
    train_data_dict, train_labels = create_data_dict(formatted_train,theta_origin_len)
    valid_data_dict, valid_labels = create_data_dict(formatted_valid,theta_origin_len)

    train_neg_likelihood_list = []
    valid_neg_likelihood_list = []
    for epoch in range(1,201):
        theta =lr_aux(train_data_dict, train_labels, dictionary, epoch ,rate)
        neg_log_likihood = 0
        for key in train_data_dict:
            label=train_labels[key]
            curr_example_indices = train_data_dict[key]
            product = sparse_dot(curr_example_indices ,theta)
            neg_log_likihood += -label*product+math.log(1+math.exp(product))
        train_avg_log_likelihood = neg_log_likihood/len(train_data_dict.keys())
        train_neg_likelihood_list.append(train_avg_log_likelihood)

        neg_log_likihood = 0
        for key in valid_data_dict:
            label=valid_labels[key]
            curr_example_indices = valid_data_dict[key]
            product = sparse_dot(curr_example_indices ,theta)
            neg_log_likihood += -label*product+math.log(1+math.exp(product))
        valid_avg_log_likelihood = neg_log_likihood/len(valid_data_dict.keys())
        valid_neg_likelihood_list.append(valid_avg_log_likelihood)

    
    train_neg_likihood_out = 'formatted_files/train_neg_likihood_model2.tsv'
    valid_neg_likihood_out = 'formatted_files/valid_neg_likihood_model2.tsv'


    train_neg_likihood_out = open(train_neg_likihood_out, 'w')
    train_neg_likihood_out.writelines("%s " % likelihood for likelihood in train_neg_likelihood_list)
    train_neg_likihood_out.close()

    valid_neg_likihood_out = open(valid_neg_likihood_out, 'w')
    valid_neg_likihood_out.writelines("%s " % likelihood for likelihood in valid_neg_likelihood_list)
    valid_neg_likihood_out.close()



if __name__ == '__main__':
    # formatted_train = 'formatted_files/formatted_train_out.tsv'
    # formatted_valid = 'formatted_files/formatted_validatoin_out.tsv'
    # formatted_test =  'formatted_files/formatted_test_out.tsv'
    # dict_input = 'handout/dict.txt'
    # train_out ='formatted_files/train_out.labels'
    # test_out = 'formatted_files/test_out.labels'
    # metrics_out = 'formatted_files/metric_out.txt'
    # epoch = 50


    # formatted_train = sys.argv[1]
    # formatted_valid = sys.argv[2]
    # formatted_test =  sys.argv[3]
    # dict_input = sys.argv[4]
    # train_out = sys.argv[5]
    # test_out = sys.argv[6]
    # metrics_out = sys.argv[7]
    # epoch = int(sys.argv[8])
    # main(formatted_train,formatted_valid,formatted_test,dict_input,train_out, test_out,metrics_out,epoch)


    avg_log_likelihood2()


