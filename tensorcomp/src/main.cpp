#include <iostream>

#include "tensor.h"

using namespace TensorComp;

int main(){
    std::cout << "Hola!" << std::endl;
    
    Tensor<float> t({4,4,4,4});
    std::cout << t << std::endl;

    t.initRandom();
    t.dump(std::cout);

    return 0;
}