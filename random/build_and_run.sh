cmake .. -DDISTRIBUTED=ON -DMULTITHREADED=ON
make -j
mpirun -N 2 --hostfile ../hosts.txt -x OMP_NUM_THREADS=8\
    --bind-to socket ./demo > distributed.log
diff probs.dat ../examples/probs.dat_random
diff stateVector.dat ../examples/stateVector.dat_random
