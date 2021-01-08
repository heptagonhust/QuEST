cmake .. -DDISTRIBUTED=ON -DMULTITHREADED=ON
make -j
mpirun -N 4 --hostfile ../hosts.txt -x OMP_NUM_THREADS=4\
    --bind-to socket ./demo > distributed.log
diff probs.dat ../examples/probs.dat_random
diff stateVector.dat ../examples/stateVector.dat_random
