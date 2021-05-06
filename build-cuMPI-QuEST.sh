TARGET_NODE=icu02
NCCL_SOCKET_IFNAME=ib0

rm -rf ~/demo ~/qft

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
if [[ "$?" -ne "0" ]]; then
  cd /root/QuEST-experiments/QuEST
  read 
fi
\cp -f src/libcuMPI.so /lib64/libcuMPI.so

cd ../../../../..
cd GHZ_QFT
if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j16
\cp -f qft ~
\cp -f QuEST/libQuEST.so /lib64/libQuEST.so

cd ../..
cd random
if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j16
\cp -f demo ~

cd ../../

rm -rf QuEST/src/GPU/cuMPI/build
rm -rf GHZ_QFT/build
rm -rf random/build