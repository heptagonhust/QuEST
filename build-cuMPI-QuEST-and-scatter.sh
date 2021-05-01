TARGET_NODE=icu02
NCCL_SOCKET_IFNAME=ib0

cd QuEST/src/GPU/cuMPI
if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake -DNCCL_LIBRARY=/lib64/libnccl.so \
      -DNCCL_INCLUDE_DIR=/usr/include/ \
      ..
make -j16
cp src/libcuMPI.so /lib64/libcuMPI.so

cd ../../../../..
cd GHZ_QFT
if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j16
cp qft ~
cp QuEST/libQuEST.so /lib64/libQuEST.so

cd ../..
cd random
if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j16
cp demo ~

scp /lib64/libcuMPI.so $TARGET_NODE:/lib64/libcuMPI.so
scp /lib64/libQuEST.so $TARGET_NODE:/lib64/libQuEST.so
scp ~/qft $TARGET_NODE:~
scp ~/demo $TARGET_NODE:~