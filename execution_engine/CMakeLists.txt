cmake_minimum_required(VERSION 3.10)
project(AlphaEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3 specifically
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 using the python module directly
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")

find_package(pybind11 REQUIRED)

# Sources
set(ENGINE_SOURCES
    matching_engine.cpp
    python_wrapper.cpp
)

# Create the module
pybind11_add_module(alpha_engine ${ENGINE_SOURCES})

# Include directory
target_include_directories(alpha_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
