. ./bin/init.sh
set -ex

if [ -d "./dist" ];then
    rm -rf ./dist
fi
mkdir ./dist
cd ./dist

cmake ..
make clean && make

# Demo 准备
cd ../TestDemo

rm -rf include
rm -rf models
rm -rf lib
ln -s ../include/ .
ln -s ../models/ .
ln -s ../lib/ .

rm -rf ./lib/libemotion_classification_engine.so
ln -s ../dist/libemotion_classification_engine.so ./lib/

cp -r ../bin/* ./bin/