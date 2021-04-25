if [ -d "build" ]; then 
  rm -rf build
fi
mkdir build
cd build
cmake ../../ -DGPUACCELERATED=ON -DMULTITHREADED=OFF
make -j6
mpirun -np 2./qft
diff probs.dat ../../examples/probs.dat_GHZ
diff stateVector.dat ../../examples/stateVector.dat_GHZ
