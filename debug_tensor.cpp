#include "src/j_interpreter_lib.hpp"
#include <iostream>

int main() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    // Create scalar tensors
    auto tensor_15 = JInterpreter::JTensor::scalar(15LL);
    auto tensor_3 = JInterpreter::JTensor::scalar(3LL);
    
    std::cout << "tensor_15: rank=" << tensor_15->rank() << ", size=" << tensor_15->size() << std::endl;
    std::cout << "tensor_3: rank=" << tensor_3->rank() << ", size=" << tensor_3->size() << std::endl;
    
    // Get flat data
    auto data_15 = tensor_15->get_flat<long long>();
    auto data_3 = tensor_3->get_flat<long long>();
    
    std::cout << "data_15.size()=" << data_15.size() << ", data_3.size()=" << data_3.size() << std::endl;
    
    for (size_t i = 0; i < data_15.size(); ++i) {
        std::cout << "data_15[" << i << "]=" << data_15[i] << std::endl;
    }
    
    for (size_t i = 0; i < data_3.size(); ++i) {
        std::cout << "data_3[" << i << "]=" << data_3[i] << std::endl;
    }
    
    return 0;
}
