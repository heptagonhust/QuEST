QUESTHOME=/home/yh/QuEST
NCCL_SOCKET_IFNAME=ib0

source env.sh

# cuMPI
cd $QUESTHOME/QuEST/src/GPU/cuMPI
if [ "$1" = "empty" ]; then
  if [ -d "build" ]; then 
    rm -rf build
  fi
  mkdir build
  cd build
  cmake -DNCCL_LIBRARY=/lib64/libnccl.so \
        -DNCCL_INCLUDE_DIR=/usr/include/ \
        ..
else
  cd build
fi
make -j
if [[ "$?" -ne "0" ]]; then
  cd $QUESTHOME
  read 
  exit 255
fi
\cp -f src/libcuMPI.so /lib64/libcuMPI.so

# Build for workload
cd $QUESTHOME
if [ "$1" = "empty" ]; then
  if [ -d "build" ]; then 
    rm -rf build
  fi
  mkdir build
  cd build
  cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
else
  cd build
fi
make -j
if [[ "$?" -ne "0" ]]; then
  cd $QUESTHOME
  read 
  exit 255
fi
\cp -f wl1 wl2 wl3 wl4 ~
\cp -f QuEST/libQuEST.so /lib64/libQuEST.so


cd $QUESTHOME
