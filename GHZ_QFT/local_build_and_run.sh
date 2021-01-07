cmake .. -DDISTRIBUTED=OFF -DMULTITHREADED=ON
make -j
./qft > distributed.log
diff probs.dat ../examples/probs.dat_GHZ
diff stateVector.dat ../examples/stateVector.dat_GHZ
