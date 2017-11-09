cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
 
cuda_add_executable(test_knearests knearests.h knearests.cu test_knearests.cu)
