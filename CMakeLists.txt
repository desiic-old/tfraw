#cmake version
cmake_minimum_required(VERSION 3.10)

#programme info
find_package(TensorflowCC REQUIRED)
add_executable(prog prog.cpp)
target_link_libraries(prog TensorflowCC::Shared)

#end of file