if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j6
mv demo ..