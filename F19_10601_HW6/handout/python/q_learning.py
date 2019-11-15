from environment import MountainCar
import numpy as np
import sys





def find_best_action(q_all,epsilon):
    # need to incorporate random sampling here    
    prob = np.random.rand(1)[0]
    if prob<= 1-epsilon:
        return np.argmax(q_all)
    else:
        return np.random.random_integers(0,2)




def convert_state_to_vector(state_dict,mode):  
    if mode == 'raw':
        state = np.zeros(2)
        for key ,value in state_dict.items():
            state[key] = value

    elif mode == 'tile':
        state = np.zeros(2048,)
        for key in state_dict:
            state[key] = 1

    return state




def update_weight(mode,W,b,q_all, best_action,lr,gamma,reward,curr_state,future_state):
    q = q_all[best_action]

    q_all_future = np.transpose(future_state).dot(W)+b
    W_grad = np.zeros((curr_state.shape[0],3))
    W_grad[:,best_action] = np.transpose(curr_state)
    W = W-lr*(q-(reward + gamma*max(q_all_future)))*W_grad
    b = b-lr*(q-(reward + gamma*max(q_all_future)))

    return W , b





def main( mode , episodes, max_iterations, epsilon, gamma, lr, weight_out,returns_out):
    sys = MountainCar(mode) 
    #initialize states. add one extra dimensions in row for bias term
    returns_out_list = []
    W = np.zeros((sys.state_space,3))   
    b= 0 


    for i in range(episodes):
        return_out = 0
        curr_state = sys.reset()
        curr_state = convert_state_to_vector(curr_state,mode)
        for itr in range(max_iterations):
            q_all = np.transpose(curr_state).dot(W)+b
            best_action =find_best_action(q_all,epsilon)
            future_state, reward, done = sys.step(best_action)
            future_state = convert_state_to_vector(future_state,mode)
            # print('future state', future_state)
            return_out +=reward

            W ,b= update_weight(mode,W,b,q_all, best_action,lr,gamma,reward,curr_state,future_state)
            curr_state = future_state
            if done:           
                # sys.reset()
                break

        returns_out_list.append(return_out) 

    
    #writing to the weight out file 
    flattened_W = np.ndarray.flatten(W)
    weight_out = open(weight_out, 'w')
    weight_out.writelines("%s\n" % b)
    weight_out.writelines("%s\n" % w for w in flattened_W)
    weight_out.close()

    #writing to the return out file
    returns_out = open(returns_out,'w')
    returns_out.writelines("%s\n" % r for r in returns_out_list)
    returns_out.close()

    print(W)
    print(b)
    return 



if __name__ == "__main__":
    # mode = 'raw'
    # weight_out  = 'output/weight_out.out'
    # returns_out = 'output/returns_out.out'
    # episodes = 4
    # max_iterations = 200
    # epsilon = 0.05
    # gamma  = 0.99
    # lr = 0.01

    mode = sys.argv[1]
    weight_out  = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma  = float(sys.argv[7])
    lr = float(sys.argv[8])



    main(mode , episodes, max_iterations, epsilon, gamma, lr, weight_out, returns_out)