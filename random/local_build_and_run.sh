cmake .. -DDISTRIBUTED=OFF -DMULTITHREADED=ON
make -j
./demo >local.log
diff probs.dat ../examples/probs.dat_random
diff stateVector.dat ../examples/stateVector.dat_random
