rm -rf build/*
cd build
cmake .. -DGPUACCELERATED=ON -DMULTITHREADED=OFF && make -j
./demo