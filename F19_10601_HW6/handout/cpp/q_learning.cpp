#include"environment.h"
#include <iostream> 
#include <vector> 
#include <eigen3/Eigen/Dense>
using namespace Eigen; 
using namespace std; 

struct weight{
    MatrixXd m; 
    float b;
};



VectorXf convert_state_to_vecotor(map<int,float> state_dict, string mode){
    if (mode.compare('raw')==0){
        VectorXf state(2);
            for (int i= 0; i< state_dict.size();i++){
                state[i] = state_dict[i];
        }
    }
    
    else if (mode.compare('tile')==0){
        VectorXf state(2048);
        map state_dict::iterator it; 
        for ( it = symbolTable.begin(); it != symbolTable.end(); it++ ){
            std::cout << it->first  // string (key)
              << ':'
              << it->second   // string's value 
              << std::endl ;
        }
    }
}





int main(char* mode, int episode, int max_iteraions, float epsilon, float gamma, float lr) {
    MountainCar<int> sys(mode);
    int state_space = sys.get_state_space();
    MatrixXd W(state_space,3);
    float b = 0;

    for (int i=0; i<episode; i++){
        int return_out = 0; 
        map<int,float> curr_state = sys.reset();

        for (int j =0;j<max_iteraions; j++){
            VectorXf vector_curr_state = convert_state_to_vecotor(curr_state, mode);
            MatrixXd q_all = vector_curr_state.transpose()*W + b;

        }
    }

    return 0;
}
