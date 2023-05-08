cmake_minimum_required(VERSION 3.10)

project(cfar)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

find_package(pybind11 REQUIRED)
set(pybind11_DIR /home/slam-emix/.local/lib/python3.8/site-packages/pybind11)
SET(PYBIND11_INCLUDE_DIRS ${pybind11_DIR}/include)
find_package(Eigen3 REQUIRED)

pybind11_add_module(${PROJECT_NAME} cfar.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
message("Dirs = ${PYBIND11_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
