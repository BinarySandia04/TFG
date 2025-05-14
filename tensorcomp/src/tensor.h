#pragma once

#include <initializer_list>
#include <vector>
#include <ostream>
#include <random>

namespace TensorComp {

template <typename T>
class Tensor {
    public:

    // TODO: Fer que es puguin multiplicar tensors contraientlos i després fer més algoritmes

    Tensor(std::initializer_list<int> list){
        int elements = 1;
        for(std::initializer_list<int>::iterator it = list.begin(); it != list.end(); it++){
            elements *= *it;
            dims.push_back(*it);
        }
        data = (float*) malloc(sizeof(float) * elements);
        nElements = elements;
    }
    
    ~Tensor(){
        free(data);
    }
    
    void initRandom(){
        std::random_device rd;
        std::mt19937 e2(rd());    
        std::normal_distribution<> dist(0, 1);    
        for(int i = 0; i < nElements; i++){
            data[i] = dist(e2);
        }
    }

    friend std::ostream& operator<< (std::ostream& stream, const Tensor<T>& tensor){
        stream << "[";
        for(int i = 0; i < tensor.dims.size(); i++){
            stream << tensor.dims[i];
            if(i < tensor.dims.size() - 1) stream << " ";
        }
        stream << "]";
        return stream;
    }

    void dump(std::ostream& stream){
        for(int i = 0; i < nElements; i++){
            stream << data[i] << " ";
            
            // line separators
            int p = i + 1;
            for(int k = 0; k < dims.size(); k++){
                if(p % dims[k] == 0) {
                    stream << "\n";
                    p /= dims[k];
                } else break;
            }
        }
    }

    private:
    float *data;
    int nElements;
    std::vector<int> dims;
};

};