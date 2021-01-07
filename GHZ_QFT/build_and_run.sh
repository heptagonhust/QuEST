cmake .. -DDISTRIBUTED=ON -DMULTITHREADED=ON
make -j
mpirun -N 2 --hostfile ../hosts.txt -x OMP_NUM_THREADS=8\
    --bind-to socket ./qft > distributed.log
diff probs.dat ../examples/probs.dat_GHZ
diff stateVector.dat ../examples/stateVector.dat_GHZ
