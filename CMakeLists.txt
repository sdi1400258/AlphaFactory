cmake_minimum_required(VERSION 3.15)
project(AlphaFactoryExecutionEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(alpha_engine
    execution_engine/orderbook.hpp
    execution_engine/matching_engine.cpp
    execution_engine/simulator.cpp
    execution_engine/risk_checks.cpp
)

target_include_directories(alpha_engine PUBLIC execution_engine)


