. ./bin/init.sh
set -ex

if [ -d "./dist" ];then
    rm -rf ./dist
fi
mkdir ./dist
cd ./dist

cmake ..

make clean && make