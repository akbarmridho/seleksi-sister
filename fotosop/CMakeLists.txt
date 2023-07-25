cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_STANDARD 17)

project(fotosop LANGUAGES CUDA CXX C)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

include_directories(${GTK3_INCLUDE_DIRS})
link_directories(${GTK3_LIBRARY_DIRS})

add_executable(fotosop
        lib/cvui.h
        lib/osdialog_gtk3.c
        lib/osdialog.c
        src/converter.cu
        src/loader.cpp
        main.cu
        )

target_link_libraries(fotosop ${OpenCV_LIBS} ${GTK3_LIBRARIES})

set_target_properties(fotosop PROPERTIES CUDA_ARCHITECTURES "75")

set_target_properties(fotosop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(
        ${lib/cvui.h}
        ${lib/osdialog.c}
        ${lib/osdialog.h}
        ${lib/osdialog_gtk3.c}
        PROPERTIES
        COMPILE_FLAGS "-w"
)