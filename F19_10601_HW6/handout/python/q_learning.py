from environment import MountainCar
import numpy as np




def find_best_action(q_all,epsilon):
    # need to incorporate random sampling here    
    return np.argmax(q_all)



def update_weight(W,q_all, best_action,lr,gamma,reward,curr_state,sys):
    q = q_all[best_action]
    future_state = sys.state
    future_state = np.append(future_state,1)
    q_all_future = np.transpose(future_state).dot(W)
    W_grad = np.zeros((sys.state_space+1,3))
    W_grad[:,best_action] = np.transpose(curr_state)
    # print(q)
    print(W_grad)
    W = W-lr*(q-(reward + gamma*max(q_all_future)))*W_grad*q

    return W







def main( mode , episodes, max_iterations, epsilon, gamma, lr):
    sys = MountainCar(mode) 
    #initialize states. add one extra dimensions in row for bias term
    W = np.zeros((sys.state_space+1,3))  
    for i in range(episodes):
        for itr in range(max_iterations):
            curr_state = sys.state
            curr_state = np.append(curr_state,1)
            q_all = np.transpose(curr_state).dot(W)
            best_action = np.argmax(q_all)
            _, reward, done = sys.step(best_action)

            if done:
                return W
            W = update_weight(W,q_all, best_action,lr,gamma,reward,curr_state,sys)
        print(W)
    return W





if __name__ == "__main__":
    mode = 'raw'
    episodes = 4
    max_iterations = 200
    epsilon = 0.05
    gamma  = 0.99
    lr = 0.01
    W = main(mode , episodes, max_iterations, epsilon, gamma, lr)
    print(W)