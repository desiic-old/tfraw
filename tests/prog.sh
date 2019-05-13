#!/bin/bash
#reference:
#https://github.com/FloopCZ/tensorflow_cc

#recreate build dir
rm -rf ./build
mkdir build

#build
cp -f CMakeLists.txt build
cd build
cmake ..
make
sudo chmod 777 prog

#run
echo -e "\n================================================================================";
cd ..
./build/prog

#end of file