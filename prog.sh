#!/bin/bash
rm -rf ./build
mkdir build
cp -f CMakeLists.txt build
cd build
cmake ..
make
sudo chmod 777 prog

echo -e "\n================================================================================";
cd ..
./build/prog

#end of file