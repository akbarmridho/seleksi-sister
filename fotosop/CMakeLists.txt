cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

project(fotosop LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fotosop src/lib/cvui.cu main.cu src/converter/grayscale.cu src/loader/image.cu)

target_link_libraries(fotosop ${OpenCV_LIBS})

set_target_properties(fotosop PROPERTIES CUDA_ARCHITECTURES "75")

set_target_properties(fotosop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)