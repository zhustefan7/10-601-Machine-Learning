from environment import MountainCar

def main( mode , episodes, max_iterations, epsilon, gamma, lr):
    sys = MountainCar(mode)
    for i in range(episodes):
        

    pass

if __name__ == "__main__":
    mode = 'raw'
    episodes = 4
    max_iterations = 200
    epsilon = 0.05
    gamma  = 0.99
    lr = 0.01
    main(mode , episodes, max_iterations, epsilon, gamma, lr)