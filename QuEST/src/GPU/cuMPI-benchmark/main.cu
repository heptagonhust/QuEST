#include <nccl.h>
#include <stdio.h>
#include <unistd.h>
#include "precise_timer.h"
#include "../cuMPI/src/cuMPI_runtime.h"

int myRank;                 // cuMPI comm local ranks
int nRanks;                 // total cuMPI comm ranks
int localRank;              // CUDA device ID

ncclUniqueId id;            // NCCL Unique ID
cuMPI_Comm comm;            // cuMPI comm
cudaStream_t commStream; // CUDA stream generated for each GPU
cuMPI_Comm defaultComm;          // cuMPI comm
cudaStream_t defaultCommStream;  // CUDA stream generated for each GPU
uint64_t hostHashs[10];     // host name hash in cuMPI
char hostname[1024];        // host name for identification in cuMPI
std::map<cuMPI_Comm, cudaStream_t> comm2stream;

// test P2P SendRecv method
int main() {
  cuMPI_Init(NULL, NULL);
  
  const int count = (1L << 29);
  const long long data_bytes = count * sizeof(float); // 4GB
  const int max_times = 4; // 4GB * 2 * 3 = 24GB

  float *d_send[max_times] = {}, *d_recv[max_times] = {};
  for (int i = 0; i < max_times; ++i) {
    CUDA_CHECK(cudaMalloc(&d_send[i], data_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv[i], data_bytes));
  }
  
  cuMPI_Status status;
  int peer = 1 - myRank;

  cuMPI_Comm realpipe, imagpipe;
  cuMPI_NewGlobalComm(&realpipe);
  cuMPI_NewGlobalComm(&imagpipe);

  toth::PreciseTimer timer;
  timer.start();

  cuMPI_CocurrentStart(realpipe);
  cuMPI_Sendrecv(d_send[0], count, cuMPI_FLOAT, peer, 0, d_recv[0], count, cuMPI_FLOAT, localRank, 0, realpipe, &status);
  cuMPI_CocurrentEnd(realpipe);

  cuMPI_CocurrentStart(imagpipe);
  cuMPI_Sendrecv(d_send[1], count, cuMPI_FLOAT, peer, 0, d_recv[1], count, cuMPI_FLOAT, localRank, 0, imagpipe, &status);
  cuMPI_CocurrentEnd(imagpipe);

  cudaDeviceSynchronize();

  timer.stop();
  double time = timer.milliseconds() / 1000.0;

  const int data_mibytes = (data_bytes >> 20);
  printf("Send & Recv NCCL tests\n");
  printf("Data Size Each Time:\t%12.6f MBytes\n", (double)data_mibytes);
  printf("Performed times count:\t    %d\n", max_times);
  printf("Total Time cost:\t%12.6f seconds\n", time);
  printf("Average Time cost:\t%12.6f seconds\n", time/(double)(max_times));
  printf("Average Bus width:\t%12.6f GBytes/s\n", (double)(max_times * data_mibytes / 1024)/time);

  for (int i = 0; i < max_times; ++i) {
    CUDA_CHECK(cudaFree(d_send[i]));
    CUDA_CHECK(cudaFree(d_recv[i]));
  }
  cuMPI_Finalize();
  
  return 0;
}


 /*
  cuMPI_Comm comm2;
  cuMPI_Comm_new(&comm2);
  //for (int i = 0; i < max_times; ++i) {
    // cudaStream_t tmp;
    // cudaStreamCreate(&tmp);
    // commStream = tmp;
    // cuMPI_Allreduce(d_send[i], d_recv[i], count, cuMPI_FLOAT, ncclSum, comm);

    cuMPI_Sendrecv(d_send[0], count, cuMPI_FLOAT, peer, 0, d_recv[0], count, cuMPI_FLOAT, localRank, 0, comm, &status);
        cudaStream_t tmp;
    cudaStreamCreate(&tmp);
    commStream = tmp;
    cuMPI_Sendrecv(d_send[1], count, cuMPI_FLOAT, peer, 0, d_recv[1], count, cuMPI_FLOAT, localRank, 0, comm2, &status);
    // cuMPI_Complex_Sendrecv(d_send[0], d_send[1], count, cuMPI_FLOAT, peer, 0, d_recv[1], d_recv[1], count, cuMPI_FLOAT, localRank, 0, comm, &status);
  //}
  cudaDeviceSynchronize();
  */


/*
  cuMPI_Comm realpipe, imagpipe;
  cuMPI_NewGlobalComm(&realpipe);
  cuMPI_NewGlobalComm(&imagpipe);

  toth::PreciseTimer timer;
  timer.start();

  cuMPI_CocurrentStart(realpipe);
  cuMPI_Sendrecv(d_send[0], count, cuMPI_FLOAT, peer, 0, d_recv[0], count, cuMPI_FLOAT, localRank, 0, realpipe, &status);
  cuMPI_CocurrentEnd(realpipe);

  cuMPI_CocurrentStart(imagpipe);
  cuMPI_Sendrecv(d_send[1], count, cuMPI_FLOAT, peer, 0, d_recv[1], count, cuMPI_FLOAT, localRank, 0, imagpipe, &status);
  cuMPI_CocurrentEnd(imagpipe);

  cudaDeviceSynchronize();

  timer.stop();
  double time = timer.milliseconds() / 1000.0;
*/