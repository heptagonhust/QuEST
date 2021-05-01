#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

double get_wall_time() {
  struct timeval time;
  if ( gettimeofday(&time,NULL) ) {
      //  Handle error
      return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time() {
  return (double)clock() / CLOCKS_PER_SEC;
}

#include "../cuMPI/src/cuMPI_runtime.h"

int myRank;                 // cuMPI comm local ranks
int nRanks;                 // total cuMPI comm ranks
int localRank;              // CUDA device ID

ncclUniqueId id;            // NCCL Unique ID
cuMPI_Comm comm;            // cuMPI comm
cudaStream_t defaultStream; // CUDA stream generated for each GPU
uint64_t hostHashs[10];     // host name hash in cuMPI
char hostname[1024];        // host name for identification in cuMPI

// test P2P SendRecv method
int main() {
  cuMPI_Init(NULL, NULL);
  
  const int count = (1L << 30);
  const long long data_bytes = count * sizeof(float);
  const int max_times = 1024;
  // const int verify_range = 100;

  float *h_send = (float *)malloc(data_bytes);
        // *h_recv = (float *)malloc(data_bytes);
  for (int i = 0; i < count; ++i) {
    h_send[i] = localRank;
  }

  float *d_send = NULL, *d_recv = NULL;
  CUDA_CHECK(cudaMalloc(&d_send, data_bytes));
  CUDA_CHECK(cudaMalloc(&d_recv, data_bytes));
  
  CUDA_CHECK(cudaMemcpy(d_send, h_send, data_bytes, cudaMemcpyHostToDevice));

  cuMPI_Status status;
  int peer = 1 - myRank;

  double t1 = get_wall_time();
  
  for (int times = 0; times < max_times; ++times) {
    cuMPI_Sendrecv(d_send, count, cuMPI_FLOAT, peer, 0, d_recv, count, cuMPI_FLOAT, localRank, 0, comm, &status);
    cudaStreamSynchronize(defaultStream);
  }

  double t2 = get_wall_time();

  const int data_gibytes = (data_bytes >> 30);
  printf("Send & Recv NCCL tests\n");
  printf("Data Size Each Time:\t%12.6f GBytes\n", (double)data_gibytes);
  printf("Performed times count:\t    %d\n", max_times);
  printf("Total Time cost:\t%12.6f seconds\n", t2 - t1);
  printf("Average Time cost:\t%12.6f seconds\n", (t2 - t1)/(double)(max_times));
  printf("Average Bus width:\t%12.6f TBytes/s\n", (double)(max_times * data_gibytes / 1024)/(t2 - t1));

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));
  free(h_send);
  cuMPI_Finalize();
  
  // CUDA_CHECK(cudaMemcpy(h_recv, d_recv, verify_range * sizeof(float), cudaMemcpyDeviceToHost));
  
  // for (int i = 0; i < verify_range; ++i) {
  //   // printf("[ %d ] -> [ %d ]\n", (int)(h_send[i]), (int)(h_recv[i]));
  //   assert( (int)abs((int)(h_send[i] - h_recv[i])) == 1 );
  // }

  // printf("Data is verified OK\n");

  // free(h_recv);
  return 0;
}
