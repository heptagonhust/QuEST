rm -rf build/*
cd build
cmake .. -DGPUACCELERATED=ON -DMULTITHREADED=OFF && make -j
\cp -f ../examples/ham_H12.dat .
./demo
