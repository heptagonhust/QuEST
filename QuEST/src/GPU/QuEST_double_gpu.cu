#include "QuEST_gpu_internal.h"
#include "cuMPI/src/cuMPI_runtime.h"

/********************** For cuMPI environment **********************/
int myRank;                      // cuMPI comm local ranks
int nRanks;                      // total cuMPI comm ranks
int localRank;                   // CUDA device ID

ncclUniqueId id;                 // NCCL Unique ID
cuMPI_Comm comm;                 // cuMPI comm
cudaStream_t commStream;         // CUDA stream generated for each GPU
cuMPI_Comm defaultComm;          // cuMPI comm
cudaStream_t defaultCommStream;  // CUDA stream generated for each GPU
uint64_t hostHashs[10];          // host name hash in cuMPI
char hostname[1024];             // host name for identification in cuMPI

std::map<cuMPI_Comm, cudaStream_t> comm2stream;

#define cuMPI_COMM_WORLD comm
#define cuMPI_QuEST_REAL MPI_QuEST_REAL
// #define cuMPI_MAX_AMPS_IN_MSG MPI_MAX_AMPS_IN_MSG
#define cuMPI_MAX_AMPS_IN_MSG (1LL<<28) // fine-tuned

cuMPI_Comm pipeReal[1000], pipeImag[1000];
/*******************************************************************/

//cuMPI_Init
//cuMPI_Allreduce
//cuMPI_Brcast
//cuMPI_SendRecv

#ifdef __cplusplus
extern "C" {
#endif

Complex statevec_calcInnerProduct(Qureg bra, Qureg ket) {
  // stage 1 done! (mode 1)
  // cuMPI done!

  Complex localInnerProd = statevec_calcInnerProductLocal(bra, ket);
  if (bra.numChunks == 1)
    return localInnerProd;
  
  qreal *localReal = mallocZeroRealInDevice(sizeof(qreal));
  qreal *localImag = mallocZeroRealInDevice(sizeof(qreal));

  setRealInDevice(localReal, &(localInnerProd.real));
  setRealInDevice(localImag, &(localInnerProd.imag));
  
  qreal *globalReal = mallocZeroRealInDevice(sizeof(qreal));
  qreal *globalImag = mallocZeroRealInDevice(sizeof(qreal));

  cuMPI_Allreduce(localReal, globalReal, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);
  cuMPI_Allreduce(localImag, globalImag, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);

  Complex globalInnerProd;
  globalInnerProd.real = getRealInDevice(globalReal);
  globalInnerProd.imag = getRealInDevice(globalImag);
  
  freeRealInDevice(localReal);
  freeRealInDevice(localImag);
  freeRealInDevice(globalReal);
  freeRealInDevice(globalImag);

  return globalInnerProd;
}


__global__ void statevec_calcTotalProbDistributedKernel (
  const long long int chunkSize, 
  qreal *stateVecReal, 
  qreal *stateVecImag,
  qreal *pTotal
){
  // Implemented using Kahan summation for greater accuracy at a slight floating
  //   point operation overhead. For more details see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
  // qreal pTotal=0;
  qreal y, t, c;

  long long int index;
  long long int numAmpsPerRank = chunkSize;

  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  if (thisTask>=numAmpsPerRank) return;
  // copyStateFromCurrentGPU(qureg);

  c = 0.0;
  for (index=0; index<numAmpsPerRank; index++){
    // Perform pTotal+=qureg.stateVec.real[index]*qureg.stateVec.real[index]; by Kahan
    y = stateVecReal[index]*stateVecReal[index] - c;
    t = *pTotal + y;
    // Don't change the bracketing on the following line
    c = ( t - *pTotal ) - y;
    *pTotal = t;
    // Perform pTotal+=qureg.stateVec.imag[index]*qureg.stateVec.imag[index]; by Kahan
    y = stateVecImag[index]*stateVecImag[index] - c;
    t = *pTotal + y;
    // Don't change the bracketing on the following line
    c = ( t - *pTotal ) - y;
    *pTotal = t;
  }
}

qreal statevec_calcTotalProb(Qureg qureg){
  // stage 1 done! fixed cuMPI device memory.
  // cuMPI done!

  // ~~phase 1 done! (mode 2)~~
  // gpu local is almost same with cpu local

  qreal *pTotal = mallocZeroRealInDevice(sizeof(qreal));
  qreal *allRankTotals = mallocZeroRealInDevice(sizeof(qreal));
  statevec_calcTotalProbDistributedKernel<<<1, 1>>>(
    qureg.numAmpsPerChunk,
    qureg.stateVec.real,
    qureg.stateVec.imag,
    pTotal
  );
  
  if (qureg.numChunks>1)
    cuMPI_Allreduce(pTotal, allRankTotals, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);
  else
    allRankTotals=pTotal;

  qreal ret = getRealInDevice(allRankTotals);
  freeRealInDevice(pTotal);
  freeRealInDevice(allRankTotals);
  return ret;
}

static int isChunkToSkipInFindPZero(int chunkId, long long int chunkSize, int measureQubit);
static int chunkIsUpper(int chunkId, long long int chunkSize, int targetQubit);
static int chunkIsUpperInOuterBlock(int chunkId, long long int chunkSize, int targetQubit, int numQubits);
static void getRotAngle(int chunkIsUpper, Complex *rot1, Complex *rot2, Complex alpha, Complex beta);
static int getChunkPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit);
static int getChunkOuterBlockPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit, int numQubits);
static int halfMatrixBlockFitsInChunk(long long int chunkSize, int targetQubit);
static int getChunkIdFromIndex(Qureg qureg, long long int index);

static int getChunkIdFromIndex(Qureg qureg, long long int index){
  return index/qureg.numAmpsPerChunk; // this is numAmpsPerChunk
}

int GPUExists(void){
  // stage 1 done! need to integrate to cuMPI, and explict set certain device.

  // there is nothing to do, maybe change it to CUDA API directly.
  int deviceCount, device;
  int gpuDeviceCount = 0;
  struct cudaDeviceProp properties;
  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess) deviceCount = 0;
  /* machines with no GPUs can still report one emulation device */
  for (device = 0; device < deviceCount; ++device) {
      cudaGetDeviceProperties(&properties, device);
      if (properties.major != 9999) { /* 9999 means emulation only */
          ++gpuDeviceCount;
      }
  }
  if (gpuDeviceCount) return 1;
  else return 0;
}

QuESTEnv createQuESTEnv(void) {
  // stage 1 done! ~~changed to cuMPI.~~
  // cuMPI done!
  
  // Local version is similar to cpu_local version. +yh

  if (!GPUExists()){
    printf("Trying to run GPU code with no GPU available\n");
    exit(EXIT_FAILURE);
  }

  QuESTEnv env;

  // init cuMPI environment
  // int rank, numRanks, initialized;
  int initialized;
  cuMPI_Initialized(&initialized);
  if (!initialized){

    cuMPI_Init(NULL, NULL);
    // cuMPI_Comm_size(cuMPI_COMM_WORLD, &numRanks);
    // cuMPI_Comm_rank(cuMPI_COMM_WORLD, &rank);

    env.rank=myRank;
    env.numRanks=nRanks;
    // cuAllocateOneGPUPerProcess();

  } else {

    printf("ERROR: Trying to initialize QuESTEnv multiple times. Ignoring...\n");

    // ensure env is initialised anyway, so the compiler is happy
    // cuMPI_Comm_size(cuMPI_COMM_WORLD, &numRanks);
    // cuMPI_Comm_rank(cuMPI_COMM_WORLD, &rank);
    env.rank=myRank;
    env.numRanks=nRanks;

  }

  seedQuESTDefault();

  return env;
}

void syncQuESTEnv(QuESTEnv env){
  // stage 1 done!
  // cuMPI done!
  // After computation in GPU device is done, synchronize cuMPI message. 
  cudaDeviceSynchronize();
  cuMPI_Barrier(cuMPI_COMM_WORLD);
}

int syncQuESTSuccess(int successCode){
  // stage 1 done!
  // cuMPI done!
  // nothing to do for GPU method.
  int *totalSuccess = (int *)mallocZeroVarInDevice(sizeof(int));

  int *d_successCode = (int *)mallocZeroVarInDevice(sizeof(int));
  setVarInDevice(d_successCode, &d_successCode, sizeof(int));

  // MPI_LAND logic and
  cuMPI_Allreduce(d_successCode, totalSuccess, 1, cuMPI_INT, cuMPI_MIN, cuMPI_COMM_WORLD);

  int ret = getIntInDevice(totalSuccess);
  freeVarInDevice(totalSuccess);
  freeVarInDevice(d_successCode);

  return ret;
}

void destroyQuESTEnv(QuESTEnv env){
  // stage 1 done!
  // ~~need to finalize nccl jobs~~

  cuMPI_Finalize();
}

void reportQuESTEnv(QuESTEnv env){
  // stage 1 done!
  // ~~maybe nothing to do.~~
  printf("EXECUTION ENVIRONMENT:\n");
  printf("Running locally on one node with GPU\n");
  printf("Number of ranks is %d\n", env.numRanks);
  printf("OpenMP disabled\n");
}

qreal statevec_getRealAmp(Qureg qureg, long long int index){
  // stage 1 done! need to optimized, no need to malloc new memory variable.
  // cuMPI done!
  // ~~phase 1 done! (mode 3)~~
  // direct copy from device state memory

  int chunkId = getChunkIdFromIndex(qureg, index);
  qreal *el = mallocZeroRealInDevice(sizeof(qreal));
  if (qureg.chunkId==chunkId){
      // el = qureg.stateVec.real[index-chunkId*qureg.numAmpsPerChunk];
      cudaMemcpy(el, &(qureg.stateVec.real[index-chunkId*qureg.numAmpsPerChunk]), 
        sizeof(*(qureg.stateVec.real)), cudaMemcpyDeviceToDevice);
  }
  cuMPI_Bcast(el, 1, cuMPI_QuEST_REAL, chunkId, cuMPI_COMM_WORLD);

  qreal ret = getRealInDevice(el);
  freeRealInDevice(el);
  return ret;
}

qreal statevec_getImagAmp(Qureg qureg, long long int index){
  // stage 1 done! need to optimized, no need to malloc new memory variable.
  // cuMPI done!
  // ~~phase 1 done! (mode 3)~~
  // direct copy from device state memory

  int chunkId = getChunkIdFromIndex(qureg, index);
  qreal *el = mallocZeroRealInDevice(sizeof(qreal));
  if (qureg.chunkId==chunkId){
      //el = qureg.stateVec.imag[index-chunkId*qureg.numAmpsPerChunk];
      cudaMemcpy(el, &(qureg.stateVec.imag[index-chunkId*qureg.numAmpsPerChunk]), 
        sizeof(*(qureg.stateVec.imag)), cudaMemcpyDeviceToDevice);
  }
  cuMPI_Bcast(el, 1, cuMPI_QuEST_REAL, chunkId, cuMPI_COMM_WORLD);

  qreal ret = getRealInDevice(el);
  freeRealInDevice(el);
  return ret;
}



/** Unmodified part derived from distributed cpu version **/


/** Returns whether a given chunk in position chunkId is in the upper or lower half of
  a block.
 *
 * @param[in] chunkId id of chunk in state vector
 * @param[in] chunkSize number of amps in chunk
 * @param[in] targetQubit qubit being rotated
 * @return 1: chunk is in upper half of block, 0: chunk is in lower half of block
 */
//! fix -- is this the same as isChunkToSkip?
static int chunkIsUpper(int chunkId, long long int chunkSize, int targetQubit)
{
    long long int sizeHalfBlock = 1LL << (targetQubit);
    long long int sizeBlock = sizeHalfBlock*2;
    long long int posInBlock = (chunkId*chunkSize) % sizeBlock;
    return posInBlock<sizeHalfBlock;
}

//! fix -- do with masking instead
static int chunkIsUpperInOuterBlock(int chunkId, long long int chunkSize, int targetQubit, int numQubits)
{
    long long int sizeOuterHalfBlock = 1LL << (targetQubit+numQubits);
    long long int sizeOuterBlock = sizeOuterHalfBlock*2;
    long long int posInBlock = (chunkId*chunkSize) % sizeOuterBlock;
    return posInBlock<sizeOuterHalfBlock;
}

/** Get rotation values for a given chunk
 * @param[in] chunkIsUpper 1: chunk is in upper half of block, 0: chunk is in lower half
 *
 * @param[out] rot1, rot2 rotation values to use, allocated for upper/lower such that
 * @verbatim
 stateUpper = rot1 * stateUpper + conj(rot2)  * stateLower
 @endverbatim
 * or
 * @verbatim
 stateLower = rot1 * stateUpper + conj(rot2)  * stateLower
 @endverbatim
 *
 * @param[in] alpha, beta initial rotation values
 */
static void getRotAngle(int chunkIsUpper, Complex *rot1, Complex *rot2, Complex alpha, Complex beta)
{
    if (chunkIsUpper){
        *rot1=alpha;
        rot2->real=-beta.real;
        rot2->imag=-beta.imag;
    } else {
        *rot1=beta;
        *rot2=alpha;
    }
}

/** Get rotation values for a given chunk given a unitary matrix
 * @param[in] chunkIsUpper 1: chunk is in upper half of block, 0: chunk is in lower half
 *
 * @param[out] rot1, rot2 rotation values to use, allocated for upper/lower such that
 * @verbatim
 stateUpper = rot1 * stateUpper + conj(rot2)  * stateLower
 @endverbatim
 * or
 * @verbatim
 stateLower = rot1 * stateUpper + conj(rot2)  * stateLower
 @endverbatim
 * @param[in] u unitary matrix operation
 */
static void getRotAngleFromUnitaryMatrix(int chunkIsUpper, Complex *rot1, Complex *rot2, ComplexMatrix2 u)
{
    if (chunkIsUpper){
        *rot1=(Complex) {.real=u.real[0][0], .imag=u.imag[0][0]};
        *rot2=(Complex) {.real=u.real[0][1], .imag=u.imag[0][1]};
    } else {
        *rot1=(Complex) {.real=u.real[1][0], .imag=u.imag[1][0]};
        *rot2=(Complex) {.real=u.real[1][1], .imag=u.imag[1][1]};
    }
}

/** get position of corresponding chunk, holding values required to
 * update values in my chunk (with chunkId) when rotating targetQubit.
 *
 * @param[in] chunkIsUpper 1: chunk is in upper half of block, 0: chunk is in lower half
 * @param[in] chunkId id of chunk in state vector
 * @param[in] chunkSize number of amps in chunk
 * @param[in] targetQubit qubit being rotated
 * @return chunkId of chunk required to rotate targetQubit
 */
static int getChunkPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit)
{
    long long int sizeHalfBlock = 1LL << (targetQubit);
    int chunksPerHalfBlock = sizeHalfBlock/chunkSize;
    if (chunkIsUpper){
        return chunkId + chunksPerHalfBlock;
    } else {
        return chunkId - chunksPerHalfBlock;
    }
}

static int getChunkOuterBlockPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit, int numQubits)
{
    long long int sizeOuterHalfBlock = 1LL << (targetQubit+numQubits);
    int chunksPerOuterHalfBlock = sizeOuterHalfBlock/chunkSize;
    if (chunkIsUpper){
        return chunkId + chunksPerOuterHalfBlock;
    } else {
        return chunkId - chunksPerOuterHalfBlock;
    }
}

static int getChunkOuterBlockPairIdForPart3(int chunkIsUpperSmallerQubit, int chunkIsUpperBiggerQubit, int chunkId,
        long long int chunkSize, int smallerQubit, int biggerQubit, int numQubits)
{
    long long int sizeOuterHalfBlockBiggerQubit = 1LL << (biggerQubit+numQubits);
    long long int sizeOuterHalfBlockSmallerQubit = 1LL << (smallerQubit+numQubits);
    int chunksPerOuterHalfBlockSmallerQubit = sizeOuterHalfBlockSmallerQubit/chunkSize;
    int chunksPerOuterHalfBlockBiggerQubit = sizeOuterHalfBlockBiggerQubit/chunkSize;
    int rank;
    if (chunkIsUpperBiggerQubit){
        rank = chunkId + chunksPerOuterHalfBlockBiggerQubit;
    } else {
        rank = chunkId - chunksPerOuterHalfBlockBiggerQubit;
    }

    if (chunkIsUpperSmallerQubit){
        rank = rank + chunksPerOuterHalfBlockSmallerQubit;
    } else {
        rank = rank - chunksPerOuterHalfBlockSmallerQubit;
    }

    return rank;
}

/** return whether the current qubit rotation will use
 * blocks that fit within a single chunk.
 *
 * @param[in] chunkSize number of amps in chunk
 * @param[in] targetQubit qubit being rotated
 * @return 1: one chunk fits in one block 0: chunk is larger than block
 */
//! fix -- this should be renamed to matrixBlockFitsInChunk
static int halfMatrixBlockFitsInChunk(long long int chunkSize, int targetQubit)
{
    long long int sizeHalfBlock = 1LL << (targetQubit);
    if (chunkSize > sizeHalfBlock) return 1;
    else return 0;
}

static int densityMatrixBlockFitsInChunk(long long int chunkSize, int numQubits, int targetQubit) {
    long long int sizeOuterHalfBlock = 1LL << (targetQubit+numQubits);
    if (chunkSize > sizeOuterHalfBlock) return 1;
    else return 0;
}



void exchangeStateVectors(Qureg qureg, int pairRank){
  // stage 1 done!
  // cuMPI done!

  // cuMPI send/receive vars
  
  int TAG=100;
  cuMPI_Status status;

  // Multiple messages are required as cuMPI uses int rather than long long int for count
  // For openmpi, messages are further restricted to 2GB in size -- do this for all cases
  // to be safe
  long long int maxMessageCount = cuMPI_MAX_AMPS_IN_MSG;
  if (qureg.numAmpsPerChunk < maxMessageCount)
      maxMessageCount = qureg.numAmpsPerChunk;

  // safely assume cuMPI_MAX... = 2^n, so division always exact
  int numMessages = qureg.numAmpsPerChunk/maxMessageCount;
  int i;
  long long int offset;
  // send my state vector to pairRank's qureg.pairStateVec
  // receive pairRank's state vector into qureg.pairStateVec
  for (i=0; i<numMessages; i++){
      offset = i*maxMessageCount;
      cuMPI_CocurrentStart(pipeReal[i]);
      cuMPI_Sendrecv(&qureg.stateVec.real[offset], maxMessageCount, cuMPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.real[offset], maxMessageCount, cuMPI_QuEST_REAL,
              pairRank, TAG, pipeReal[i], &status);
      cuMPI_CocurrentEnd(pipeReal[i]);
      //printf("rank: %d err: %d\n", qureg.rank, err);
      cuMPI_CocurrentStart(pipeImag[i]);
      cuMPI_Sendrecv(&qureg.stateVec.imag[offset], maxMessageCount, cuMPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.imag[offset], maxMessageCount, cuMPI_QuEST_REAL,
              pairRank, TAG, pipeImag[i], &status);
      cuMPI_CocurrentEnd(pipeImag[i]);
  }

  for (i=0; i<numMessages; i++) {
    cudaStreamSynchronize(comm2stream[pipeReal[i]]);
    cudaStreamSynchronize(comm2stream[pipeImag[i]]);
  }
}

void exchangePairStateVectorHalves(Qureg qureg, int pairRank){
  // stage 1 done!
  // cuMPI done!

  // cuMPI send/receive vars
  int TAG=100;
  cuMPI_Status status;
  long long int numAmpsToSend = qureg.numAmpsPerChunk >> 1;

  // Multiple messages are required as cuMPI uses int rather than long long int for count
  // For openmpi, messages are further restricted to 2GB in size -- do this for all cases
  // to be safe
  long long int maxMessageCount = cuMPI_MAX_AMPS_IN_MSG;
  if (numAmpsToSend < maxMessageCount)
      maxMessageCount = numAmpsToSend;

  // safely assume cuMPI_MAX... = 2^n, so division always exact
  int numMessages = numAmpsToSend/maxMessageCount;
  int i;
  long long int offset;
  // send the bottom half of my state vector to the top half of pairRank's qureg.pairStateVec
  // receive pairRank's state vector into the top of qureg.pairStateVec
  for (i=0; i<numMessages; i++){
      offset = i*maxMessageCount;
      cuMPI_CocurrentStart(pipeReal[i]);
      cuMPI_Sendrecv(&qureg.pairStateVec.real[offset+numAmpsToSend], maxMessageCount,
              cuMPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.real[offset], maxMessageCount, cuMPI_QuEST_REAL,
              pairRank, TAG, pipeReal[i], &status);
      cuMPI_CocurrentEnd(pipeReal[i]);

      //printf("rank: %d err: %d\n", qureg.rank, err);
      cuMPI_CocurrentStart(pipeImag[i]);
      cuMPI_Sendrecv(&qureg.pairStateVec.imag[offset+numAmpsToSend], maxMessageCount,
              cuMPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.imag[offset], maxMessageCount, cuMPI_QuEST_REAL,
              pairRank, TAG, pipeImag[i], &status);
      cuMPI_CocurrentEnd(pipeImag[i]);
  }
  for (i=0; i<numMessages; i++) {
    cudaStreamSynchronize(comm2stream[pipeReal[i]]);
    cudaStreamSynchronize(comm2stream[pipeImag[i]]);
  }
}

//TODO -- decide where this function should go. It is a preparation for MPI data transfer function
void compressPairVectorForSingleQubitDepolarise(Qureg qureg, const int targetQubit){
  long long int sizeInnerBlock, sizeInnerHalfBlock;
  long long int sizeOuterColumn, sizeOuterHalfColumn;
  long long int thisInnerBlock, // current block
       thisOuterColumn, // current column in density matrix
       thisIndex,    // current index in (density matrix representation) state vector
       thisIndexInOuterColumn,
       thisIndexInInnerBlock;

  int outerBit;

  long long int thisTask;
  const long long int numTasks=qureg.numAmpsPerChunk>>1;

  // set dimensions
  sizeInnerHalfBlock = 1LL << targetQubit;
  sizeInnerBlock     = 2LL * sizeInnerHalfBlock;
  sizeOuterHalfColumn = 1LL << qureg.numQubitsRepresented;
  sizeOuterColumn     = 2LL * sizeOuterHalfColumn;

# ifdef _OPENMP
# pragma omp parallel \
  shared   (sizeInnerBlock,sizeInnerHalfBlock,sizeOuterColumn,sizeOuterHalfColumn,qureg) \
  private  (thisTask,thisInnerBlock,thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
              thisIndexInInnerBlock,outerBit)
# endif
  {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
      // thisTask iterates over half the elements in this process' chunk of the density matrix
      // treat this as iterating over all columns, then iterating over half the values
      // within one column.
      // If this function has been called, this process' chunk contains half an
      // outer block or less
      for (thisTask=0; thisTask<numTasks; thisTask++) {
          // we want to process all columns in the density matrix,
          // updating the values for half of each column (one half of each inner block)
          thisOuterColumn = thisTask / sizeOuterHalfColumn;
          thisIndexInOuterColumn = thisTask&(sizeOuterHalfColumn-1); // thisTask % sizeOuterHalfColumn
          thisInnerBlock = thisIndexInOuterColumn/sizeInnerHalfBlock;
          // get index in state vector corresponding to upper inner block
          thisIndexInInnerBlock = thisTask&(sizeInnerHalfBlock-1); // thisTask % sizeInnerHalfBlock
          thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlock*sizeInnerBlock
              + thisIndexInInnerBlock;
          // check if we are in the upper or lower half of an outer block
          outerBit = extractBitOnCPU(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
          // if we are in the lower half of an outer block, shift to be in the lower half
          // of the inner block as well (we want to dephase |0><0| and |1><1| only)
          thisIndex += outerBit*(sizeInnerHalfBlock);

          // NOTE: at this point thisIndex should be the index of the element we want to
          // dephase in the chunk of the state vector on this process, in the
          // density matrix representation.
          // thisTask is the index of the pair element in pairStateVec
          // we will populate the second half of pairStateVec with this process'
          // data to send

          qureg.pairStateVec.real[thisTask+numTasks] = qureg.stateVec.real[thisIndex];
          qureg.pairStateVec.imag[thisTask+numTasks] = qureg.stateVec.imag[thisIndex];

      }
  }
}

__global__ void statevec_compactUnitaryDistributedKernel (
  const long long int chunkSize,
  Complex rot1, Complex rot2,
  ComplexArray deviceStateVecUp,
  ComplexArray deviceStateVecLo,
  ComplexArray deviceStateVecOut
){
  qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
  qreal rot2Real=rot2.real, rot2Imag=rot2.imag;
  qreal *stateVecRealUp=deviceStateVecUp.real, *stateVecImagUp=deviceStateVecUp.imag;
  qreal *stateVecRealLo=deviceStateVecLo.real, *stateVecImagLo=deviceStateVecLo.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;

  // store current state vector values in temp variables
  stateRealUp = stateVecRealUp[thisTask];
  stateImagUp = stateVecImagUp[thisTask];

  stateRealLo = stateVecRealLo[thisTask];
  stateImagLo = stateVecImagLo[thisTask];

  // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
  stateVecRealOut[thisTask] = rot1Real*stateRealUp - rot1Imag*stateImagUp + rot2Real*stateRealLo + rot2Imag*stateImagLo;
  stateVecImagOut[thisTask] = rot1Real*stateImagUp + rot1Imag*stateRealUp + rot2Real*stateImagLo - rot2Imag*stateRealLo;
}

/** Rotate a single qubit in the state vector of probability amplitudes, 
 * given two complex numbers alpha and beta, 
 * and a subset of the state vector with upper and lower block values stored seperately.
 *                                                                       
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
 void statevec_compactUnitaryDistributed (Qureg qureg,
  Complex rot1, Complex rot2,
  ComplexArray stateVecUp,
  ComplexArray stateVecLo,
  ComplexArray stateVecOut)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_compactUnitaryDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk,
    rot1,
    rot2,
    stateVecUp, //upper
    stateVecLo, //lower
    stateVecOut); //output
}

void statevec_compactUnitary(Qureg qureg, const int targetQubit, Complex alpha, Complex beta)
{
  // stage 1 done! need to optimize!
  // !!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  Complex rot1, rot2;

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_compactUnitaryLocal(qureg, targetQubit, alpha, beta);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    getRotAngle(rankIsUpper, &rot1, &rot2, alpha, beta);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);

    // this rank's values are either in the upper of lower half of the block.
    // send values to compactUnitaryDistributed in the correct order
    if (rankIsUpper){
      statevec_compactUnitaryDistributed(qureg,rot1,rot2,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec); //output
    } else {
      statevec_compactUnitaryDistributed(qureg,rot1,rot2,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec); //output
    }
  }
}

void statevec_unitary(Qureg qureg, const int targetQubit, ComplexMatrix2 u)
{
  // !!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  Complex rot1, rot2;

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_unitaryLocal(qureg, targetQubit, u);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    getRotAngleFromUnitaryMatrix(rankIsUpper, &rot1, &rot2, u);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);

    // this rank's values are either in the upper of lower half of the block.
    // send values to compactUnitaryDistributed in the correct order
    if (rankIsUpper){
      statevec_unitaryDistributed(qureg,rot1,rot2,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec); //output
    } else {
      statevec_unitaryDistributed(qureg,rot1,rot2,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec); //output
    }
  }
}

__global__ void statevec_controlledCompactUnitaryDistributedKernel (
  const long long int chunkSize,
  const long long int chunkId,
  const int controlQubit,
  Complex rot1, Complex rot2,
  ComplexArray deviceStateVecUp,
  ComplexArray deviceStateVecLo,
  ComplexArray deviceStateVecOut
){
  qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  // const long long int chunkSize=qureg.numAmpsPerChunk;
  // const long long int chunkId=qureg.chunkId;

  qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
  qreal rot2Real=rot2.real, rot2Imag=rot2.imag;
  qreal *stateVecRealUp=deviceStateVecUp.real, *stateVecImagUp=deviceStateVecUp.imag;
  qreal *stateVecRealLo=deviceStateVecLo.real, *stateVecImagLo=deviceStateVecLo.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;

  int controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
  if (controlBit){
      // store current state vector values in temp variables
      stateRealUp = stateVecRealUp[thisTask];
      stateImagUp = stateVecImagUp[thisTask];

      stateRealLo = stateVecRealLo[thisTask];
      stateImagLo = stateVecImagLo[thisTask];

      // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
      stateVecRealOut[thisTask] = rot1Real*stateRealUp - rot1Imag*stateImagUp + rot2Real*stateRealLo + rot2Imag*stateImagLo;
      stateVecImagOut[thisTask] = rot1Real*stateImagUp + rot1Imag*stateRealUp + rot2Real*stateImagLo - rot2Imag*stateRealLo;
  }
}

/** Rotate a single qubit in the state vector of probability amplitudes, given two complex 
 * numbers alpha and beta and a subset of the state vector with upper and lower block values 
 * stored seperately. Only perform the rotation where the control qubit is one.
 *                                               
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] controlQubit qubit to determine whether or not to perform a rotation 
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
 void statevec_controlledCompactUnitaryDistributed (Qureg qureg, const int controlQubit,
  Complex rot1, Complex rot2,
  ComplexArray stateVecUp,
  ComplexArray stateVecLo,
  ComplexArray stateVecOut)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_controlledCompactUnitaryDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk,
    qureg.chunkId,
    controlQubit,
    rot1,
    rot2, 
    stateVecUp, 
    stateVecLo,
    stateVecOut
  );
}

void statevec_controlledCompactUnitary(Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta)
{
  // stage 1 done! need to optimize!

  //!!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  Complex rot1, rot2;

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_controlledCompactUnitaryLocal(qureg, controlQubit, targetQubit, alpha, beta);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    getRotAngle(rankIsUpper, &rot1, &rot2, alpha, beta);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    //printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);

    // this rank's values are either in the upper of lower half of the block. send values to controlledCompactUnitaryDistributed
    // in the correct order
    if (rankIsUpper){
      statevec_controlledCompactUnitaryDistributed(qureg,controlQubit,rot1,rot2,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec); //output
    } else {
      statevec_controlledCompactUnitaryDistributed(qureg,controlQubit,rot1,rot2,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec); //output
    }
  }
}

void statevec_controlledUnitary(Qureg qureg, const int controlQubit, const int targetQubit,
  ComplexMatrix2 u)
{

  //!! simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  Complex rot1, rot2;

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_controlledUnitaryLocal(qureg, controlQubit, targetQubit, u);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    getRotAngleFromUnitaryMatrix(rankIsUpper, &rot1, &rot2, u);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    //printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);

    // this rank's values are either in the upper of lower half of the block. send values to controlledUnitaryDistributed
    // in the correct order
    if (rankIsUpper){
      statevec_controlledUnitaryDistributed(qureg,controlQubit,rot1,rot2,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec); //output
    } else {
      statevec_controlledUnitaryDistributed(qureg,controlQubit,rot1,rot2,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec); //output
    }
  }
}

void statevec_multiControlledUnitary(Qureg qureg, long long int ctrlQubitsMask, long long int ctrlFlipMask, const int targetQubit, ComplexMatrix2 u)
{

  //!!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  Complex rot1, rot2;

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_multiControlledUnitaryLocal(qureg, targetQubit, ctrlQubitsMask, ctrlFlipMask, u);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    getRotAngleFromUnitaryMatrix(rankIsUpper, &rot1, &rot2, u);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);

    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);

    // this rank's values are either in the upper of lower half of the block. send values to multiControlledUnitaryDistributed
    // in the correct order
    if (rankIsUpper){
      statevec_multiControlledUnitaryDistributed(qureg,targetQubit,ctrlQubitsMask,ctrlFlipMask,rot1,rot2,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec); //output
    } else {
      statevec_multiControlledUnitaryDistributed(qureg,targetQubit,ctrlQubitsMask,ctrlFlipMask,rot1,rot2,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec); //output
    }
  }
}

__global__ void statevec_pauliXDistributedKernel(
  const long long int chunkSize,
  ComplexArray deviceStateVecIn,
  ComplexArray deviceStateVecOut
){
  // stage 1 done! need to optimize!

  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  qreal *stateVecRealIn=deviceStateVecIn.real, *stateVecImagIn=deviceStateVecIn.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;

  stateVecRealOut[thisTask] = stateVecRealIn[thisTask];
  stateVecImagOut[thisTask] = stateVecImagIn[thisTask];
}

/** Rotate a single qubit by {{0,1},{1,0}.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *
 *  @remarks Qubits are zero-based and the
 *  the first qubit is the rightmost
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_pauliXDistributed (Qureg qureg,
        ComplexArray stateVecIn,
        ComplexArray stateVecOut)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_pauliXDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg.numAmpsPerChunk, stateVecIn, stateVecOut);
}

void statevec_pauliX(Qureg qureg, const int targetQubit)
{

  // stage 1 done!

  //!!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_pauliXLocal(qureg, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    //printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block. pauliX just replaces
    // this rank's values with pair values
    statevec_pauliXDistributed(qureg,
            qureg.pairStateVec, // in
            qureg.stateVec/*not sure TODO*/); // out
  }
}


__global__ void statevec_controlledNotDistributedKernel (
  const long long int chunkSize, 
  const long long int chunkId,
  const int controlQubit,
  ComplexArray deviceStateVecIn,
  ComplexArray deviceStateVecOut
){
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  // const long long int chunkSize=qureg.numAmpsPerChunk;
  // const long long int chunkId=qureg.chunkId;
  
  int controlBit;
  
  qreal *stateVecRealIn=deviceStateVecIn.real, *stateVecImagIn=deviceStateVecIn.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;
  
  controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
  if (controlBit){
      stateVecRealOut[thisTask] = stateVecRealIn[thisTask];
      stateVecImagOut[thisTask] = stateVecImagIn[thisTask];
  }
}

/** Rotate a single qubit by {{0,1},{1,0}.
 *  Operate on a subset of the state vector with upper and lower block values 
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk. Only perform the rotation
 *  for elements where controlQubit is one.
 *                                          
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
 void statevec_controlledNotDistributed (Qureg qureg, const int controlQubit,
  ComplexArray stateVecIn,
  ComplexArray stateVecOut)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_controlledNotDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk, 
    qureg.chunkId, 
    controlQubit, 
    stateVecIn, 
    stateVecOut
  );  
} 


void statevec_controlledNot(Qureg qureg, const int controlQubit, const int targetQubit)
{
  // stage 1 done! need to optimize!

  //!!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  int rankIsUpper; 	// rank's chunk is in upper half of block
  int pairRank; 		// rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_controlledNotLocal(qureg, controlQubit, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block
    if (rankIsUpper){
      statevec_controlledNotDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec); //out
    } else {
      statevec_controlledNotDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec); //out
    }
  // displayDeviceVarOnHost(qureg.stateVec.real, (qureg.stateVec.real + qureg.numAmpsPerChunk));

  }
  // displayDeviceVarOnHost(qureg.stateVec.real, (qureg.stateVec.real + qureg.numAmpsPerChunk));

}

__global__ void statevec_pauliYDistributedKernel(
  const long long int chunkSize,
  ComplexArray deviceStateVecIn,
  ComplexArray deviceStateVecOut,
  int updateUpper, const int conjFac
){
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  qreal *stateVecRealIn=deviceStateVecIn.real, *stateVecImagIn=deviceStateVecIn.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;

  int realSign=1, imagSign=1;
  if (updateUpper) imagSign=-1;
  else realSign = -1;

  stateVecRealOut[thisTask] = conjFac * realSign * stateVecImagIn[thisTask];
  stateVecImagOut[thisTask] = conjFac * imagSign * stateVecRealIn[thisTask];
}

/** Rotate a single qubit by +-{{0,-i},{i,0}.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *
 *  @remarks Qubits are zero-based and the
 *  the first qubit is the rightmost
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[in] updateUpper flag, 1: updating upper values, 0: updating lower values in block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_pauliYDistributed(Qureg qureg,
        ComplexArray stateVecIn,
        ComplexArray stateVecOut,
        int updateUpper, const int conjFac)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_pauliYDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk, 
    stateVecIn, 
    stateVecOut, 
    updateUpper, 
    conjFac
  );
}

void statevec_pauliY(Qureg qureg, const int targetQubit)
{
  // stage 1 done! need to optizize!

  //!!care about local_cpu code blow:
  //int conjFac = 1;

	int conjFac = 1;

    // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
    int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
    int rankIsUpper;	// rank's chunk is in upper half of block
    int pairRank; 		// rank of corresponding chunk

    if (useLocalDataOnly){
        statevec_pauliYLocal(qureg, targetQubit); // local gpu version separated pauliY & pauliYConj
    } else {
        // need to get corresponding chunk of state vector from other rank
        rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
        pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
        // get corresponding values from my pair
        exchangeStateVectors(qureg, pairRank);
        // this rank's values are either in the upper of lower half of the block
        statevec_pauliYDistributed(qureg,
                qureg.pairStateVec, // in
                /*not sure TODO*/qureg.stateVec, // out
                rankIsUpper, conjFac);
    }
}

void statevec_pauliYConj(Qureg qureg, const int targetQubit)
{
  // stage 1 done! need to optizize!

  //!!similar to pauliY, care code from cpu_local:
  //int conjFac = -1;

	int conjFac = -1;

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  int rankIsUpper;	// rank's chunk is in upper half of block
  int pairRank; 		// rank of corresponding chunk

  if (useLocalDataOnly){
    statevec_pauliYConjLocal(qureg, targetQubit); // local gpu version separated pauliY & pauliYConj
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block
    statevec_pauliYDistributed(qureg,
            qureg.pairStateVec, // in
            qureg.stateVec/*not sure TODO*/, // out
            rankIsUpper, conjFac);
  }
}

__global__ void statevec_controlledPauliYDistributedKernel (
  const long long int chunkSize, 
  const long long int chunkId,
  const int controlQubit,
  ComplexArray deviceStateVecIn,
  ComplexArray deviceStateVecOut, 
  const int conjFac
){
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x; 
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  // const long long int chunkSize=qureg.numAmpsPerChunk;
  // const long long int chunkId=qureg.chunkId;
  
  qreal *stateVecRealIn=deviceStateVecIn.real, *stateVecImagIn=deviceStateVecIn.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;
  
  int controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
  if (controlBit){
      stateVecRealOut[thisTask] = conjFac * stateVecImagIn[thisTask];
      stateVecImagOut[thisTask] = conjFac * -stateVecRealIn[thisTask];
  }
}

void statevec_controlledPauliYDistributed (Qureg qureg, const int controlQubit,
  ComplexArray stateVecIn,
  ComplexArray stateVecOut, const int conjFac)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_controlledPauliYDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk, 
    qureg.chunkId,
    controlQubit,
    stateVecIn, 
    stateVecOut,
    conjFac
  );
} 

void statevec_controlledPauliY(Qureg qureg, const int controlQubit, const int targetQubit)
{
  // stage 1 done! need to optimize!

  // 20210420, conjLocal & Local can merge into one in local gpu!
  //!!care about code of local_cpu below:
  //int conjFac = 1;

	int conjFac = 1;

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  int rankIsUpper; 	// rank's chunk is in upper half of block
  int pairRank; 		// rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_controlledPauliYLocal(qureg, controlQubit, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block
    if (rankIsUpper){
      statevec_controlledPauliYDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec,
              conjFac); //out
    } else {
      statevec_controlledPauliYDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec,
              -conjFac); //out
    }
  }
}

void statevec_controlledPauliYConj(Qureg qureg, const int controlQubit, const int targetQubit)
{
  // stage 1 done! need to optimize!

  // 20210420, conjLocal & Local can merge into one in local gpu!
  //!!care about the code of local_cpu blow:
  //int conjFac = -1;

	int conjFac = -1;

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);
  int rankIsUpper; 	// rank's chunk is in upper half of block
  int pairRank; 		// rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_controlledPauliYConjLocal(qureg, controlQubit, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block
    if (rankIsUpper){
      statevec_controlledPauliYDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec,
              conjFac); //out
    } else {
      statevec_controlledPauliYDistributed(qureg,controlQubit,
              qureg.pairStateVec, //in
              qureg.stateVec,
              -conjFac); //out
    }
  }
}

__global__ void statevec_hadamardDistributedKernel(
  const long long int chunkSize,
  ComplexArray deviceStateVecUp,
  ComplexArray deviceStateVecLo,
  ComplexArray deviceStateVecOut,
  int updateUpper
){
  qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
  long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  const long long int numTasks = chunkSize;

  if (thisTask>=numTasks) return;

  int sign;
  if (updateUpper) sign=1;
  else sign=-1;

  qreal recRoot2 = 1.0/sqrt(2.0);

  qreal *stateVecRealUp=deviceStateVecUp.real, *stateVecImagUp=deviceStateVecUp.imag;
  qreal *stateVecRealLo=deviceStateVecLo.real, *stateVecImagLo=deviceStateVecLo.imag;
  qreal *stateVecRealOut=deviceStateVecOut.real, *stateVecImagOut=deviceStateVecOut.imag;

  // store current state vector values in temp variables
  stateRealUp = stateVecRealUp[thisTask];
  stateImagUp = stateVecImagUp[thisTask];

  stateRealLo = stateVecRealLo[thisTask];
  stateImagLo = stateVecImagLo[thisTask];

  stateVecRealOut[thisTask] = recRoot2*(stateRealUp + sign*stateRealLo);
  stateVecImagOut[thisTask] = recRoot2*(stateImagUp + sign*stateImagLo);
}

/** Rotate a single qubit by {{1,1},{1,-1}}/sqrt2.
 *  Operate on a subset of the state vector with upper and lower block values 
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *                                          
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[in] updateUpper flag, 1: updating upper values, 0: updating lower values in block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
 void statevec_hadamardDistributed(Qureg qureg,
  ComplexArray stateVecUp,
  ComplexArray stateVecLo,
  ComplexArray stateVecOut,
  int updateUpper)
{
  assert(isReadyOnGPU(qureg));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_hadamardDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk,
    stateVecUp, //upper
    stateVecLo, //lower
    stateVecOut, updateUpper //output
  );
}

void statevec_hadamard(Qureg qureg, const int targetQubit)
{
  // stage 1 done! need to optimize!
  //!!simple in local_cpu

  // flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
  int useLocalDataOnly = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targetQubit);

  // rank's chunk is in upper half of block
  int rankIsUpper;
  int pairRank; // rank of corresponding chunk

  if (useLocalDataOnly){
    // all values required to update state vector lie in this rank
    statevec_hadamardLocal(qureg, targetQubit);
  } else {
    // need to get corresponding chunk of state vector from other rank
    rankIsUpper = chunkIsUpper(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    pairRank = getChunkPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk, targetQubit);
    //printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
    // get corresponding values from my pair
    exchangeStateVectors(qureg, pairRank);
    // this rank's values are either in the upper of lower half of the block. send values to hadamardDistributed
    // in the correct order
    if (rankIsUpper){
      statevec_hadamardDistributed(qureg,
              qureg.stateVec, //upper
              qureg.pairStateVec, //lower
              qureg.stateVec, rankIsUpper); //output
    } else {
      statevec_hadamardDistributed(qureg,
              qureg.pairStateVec, //upper
              qureg.stateVec, //lower
              qureg.stateVec, rankIsUpper); //output
    }
  }
  // displayDeviceVarOnHost(qureg.stateVec.real, (qureg.stateVec.real + qureg.numAmpsPerChunk));
}

struct calcProb{
  const qreal *_stateVecReal;
  const qreal *_stateVecImag;
  calcProb(qreal *stateVecReal, qreal *stateVecImag) : _stateVecReal(stateVecReal), _stateVecImag(stateVecImag) { }
  __host__ __device__        
  qreal operator()(const int& idx) const { 
    return _stateVecReal[idx] * _stateVecReal[idx] +
           _stateVecImag[idx] * _stateVecImag[idx];
  }
};

// __global__ void statevec_findProbabilityOfZeroDistributedKernel (
//   const long long int chunkSize,
//   qreal *stateVecReal,
//   qreal *stateVecImag,
//   qreal *totalProbability // ----- measured probability
// ) {
//   // ----- temp variables
//   long long int thisTask = blockIdx.x*blockDim.x + threadIdx.x; // task based approach for expose loop with small granularity
//   const long long int numTasks = chunkSize;
//   if (thisTask>=numTasks) return;

//   // ---------------------------------------------------------------- //
//   //            find probability                                      //
//   // ---------------------------------------------------------------- //

//   atomicAdd(totalProbability, stateVecReal[thisTask]*stateVecReal[thisTask]
//           + stateVecImag[thisTask]*stateVecImag[thisTask]);
// }

/** Measure the probability of a specified qubit being in the zero state across all amplitudes held in this chunk.
 * Size of regions to skip is a multiple of chunkSize.
 * The results are communicated and aggregated by the caller
 *  
 *  @param[in] qureg object representing the set of qubits
 *  @return probability of qubit measureQubit being zero
 */
 qreal statevec_findProbabilityOfZeroDistributed (Qureg qureg) {
  // stage 1 done!
  qreal totalProbability = 0.0;
  // int threadsPerCUDABlock, CUDABlocks;
  // threadsPerCUDABlock = 128;
  // CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  // statevec_findProbabilityOfZeroDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
  //   qureg.numAmpsPerChunk,
  //   qureg.stateVec.real,
  //   qureg.stateVec.imag,
  //   &totalProbability
  // );

  thrust::device_vector<int> d_idx(qureg.numAmpsPerChunk);
  thrust::sequence(d_idx.begin(), d_idx.end());
  calcProb unary_op(qureg.stateVec.real, qureg.stateVec.imag);
  thrust::plus<qreal> binary_op;
  qreal init = 0.0;
  totalProbability = thrust::transform_reduce(d_idx.begin(), d_idx.end(), unary_op, init, binary_op);
  return totalProbability;
}


/** Find chunks to skip when calculating probability of qubit being zero.
 * When calculating probability of a bit q being zero,
 * sum up 2^q values, then skip 2^q values, etc. This function finds if an entire chunk
 * is in the range of values to be skipped
 *
 * @param[in] chunkId id of chunk in state vector
 * @param[in] chunkSize number of amps in chunk
 * @param[in] measureQubi qubit being measured
 * @return int -- 1: skip, 0: don't skip
 */
 static int isChunkToSkipInFindPZero(int chunkId, long long int chunkSize, int measureQubit)
 {
     long long int sizeHalfBlock = 1LL << (measureQubit);
     int numChunksToSkip = sizeHalfBlock/chunkSize;
     // calculate probability by summing over numChunksToSkip, then skipping numChunksToSkip, etc
     int bitToCheck = chunkId & numChunksToSkip;
     return bitToCheck;
 }

qreal statevec_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome)
{
  // stage 1 done! optimize statevec_findProbabilityOfZeroDistributed!
  // cuMPI done!
  //~~!!need to compare to gpu_local & cpu_local~~

  // qreal stateProb=0, totalStateProb=0;
  qreal *stateProb = mallocZeroRealInDevice(sizeof(qreal));
  qreal *totalStateProb = mallocZeroRealInDevice(sizeof(qreal));

  int skipValuesWithinRank = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, measureQubit);
  if (skipValuesWithinRank) {
    // printf("in 1486 no\n");
    qreal h_tmp = statevec_findProbabilityOfZeroLocal(qureg, measureQubit);
    setRealInDevice(stateProb, &h_tmp);
  } else {
    if (!isChunkToSkipInFindPZero(qureg.chunkId, qureg.numAmpsPerChunk, measureQubit)){
      // printf("in 1490\n");
      qreal h_tmp = statevec_findProbabilityOfZeroDistributed(qureg);
      setRealInDevice(stateProb, &h_tmp);
    } else {
      // stateProb = 0;
    }
  }
  cuMPI_Allreduce(stateProb, totalStateProb, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);

  qreal h_totalStateProb = getRealInDevice(totalStateProb);
  freeRealInDevice(stateProb);
  freeRealInDevice(totalStateProb);
  if (outcome==1) h_totalStateProb = 1.0 - h_totalStateProb;
  return h_totalStateProb;
}

void statevec_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal totalStateProb)
{
  //!!simple return in cpu_local

  int skipValuesWithinRank = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, measureQubit);
  if (skipValuesWithinRank) {
    statevec_collapseToKnownProbOutcomeLocal(qureg, measureQubit, outcome, totalStateProb);
  } else {
    if (!isChunkToSkipInFindPZero(qureg.chunkId, qureg.numAmpsPerChunk, measureQubit)){
      // chunk has amps for q=0
      if (outcome==0) statevec_collapseToKnownProbOutcomeDistributedRenorm(qureg, measureQubit,
              totalStateProb);
      else statevec_collapseToOutcomeDistributedSetZero(qureg);
    } else {
        // chunk has amps for q=1
      if (outcome==1) statevec_collapseToKnownProbOutcomeDistributedRenorm(qureg, measureQubit,
              totalStateProb);
      else statevec_collapseToOutcomeDistributedSetZero(qureg);
    }
  }
}

void seedQuESTDefault(){
  // stage 1 done!
  // cuMPI done!

  //!!cpu_local is similar to gpu_local

  // init MT random number generator with three keys -- time and pid
  // for the cuMPI version, it is ok that all procs will get the same seed as random numbers will only be
  // used by the master process

  unsigned long int key[2];
  getQuESTDefaultSeedKey(key);
  // this seed will be used to generate the same random number on all procs,
  // therefore we want to make sure all procs receive the same key
  // using cuMPI_UNSIGNED_LONG

  unsigned long int *d_key = (unsigned long int *)mallocZeroVarInDevice(2 * sizeof(unsigned long int));
  setVarInDevice(d_key, key, 2 * sizeof(unsigned long int));
  cuMPI_Bcast(d_key, 2, cuMPI_UINT32_T, 0, cuMPI_COMM_WORLD);
  freeVarInDevice(d_key);
  init_by_array(key, 2);
}

/** returns -1 if this node contains no amplitudes where qb1 and qb2
 * have opposite parity, otherwise returns the global index of one
 * of such contained amplitudes (not necessarily the first)
 */
 long long int getGlobalIndOfOddParityInChunk(Qureg qureg, int qb1, int qb2) {
  long long int chunkStartInd = qureg.numAmpsPerChunk * qureg.chunkId;
  long long int chunkEndInd = chunkStartInd + qureg.numAmpsPerChunk; // exclusive
  long long int oddParityInd;

  if (extractBitOnCPU(qb1, chunkStartInd) != extractBitOnCPU(qb2, chunkStartInd))
      return chunkStartInd;

  oddParityInd = flipBitOnCPU(chunkStartInd, qb1);
  if (oddParityInd >= chunkStartInd && oddParityInd < chunkEndInd)
      return oddParityInd;

  oddParityInd = flipBitOnCPU(chunkStartInd, qb2);
  if (oddParityInd >= chunkStartInd && oddParityInd < chunkEndInd)
      return oddParityInd;

  return -1;
}

void statevec_swapQubitAmps(Qureg qureg, int qb1, int qb2) {

  //!!simple return in cpu_local

  // perform locally if possible
  int qbBig = (qb1 > qb2)? qb1 : qb2;
  if (halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, qbBig))
      return statevec_swapQubitAmpsLocal(qureg, qb1, qb2);

  // do nothing if this node contains no amplitudes to swap
  long long int oddParityGlobalInd = getGlobalIndOfOddParityInChunk(qureg, qb1, qb2);
  if (oddParityGlobalInd == -1)
      return;

  // determine and swap amps with pair node
  int pairRank = flipBitOnCPU(flipBitOnCPU(oddParityGlobalInd, qb1), qb2) / qureg.numAmpsPerChunk;
  exchangeStateVectors(qureg, pairRank);
  statevec_swapQubitAmpsDistributed(qureg, pairRank, qb1, qb2);
}

/** This calls swapQubitAmps only when it would involve a distributed communication;
 * if the qubit chunks already fit in the node, it operates the unitary direct.
 * Note the order of q1 and q2 in the call to twoQubitUnitaryLocal is important.
 *
 * @TODO: refactor so that the 'swap back' isn't performed; instead the qubit locations
 * are updated.
 * @TODO: the double swap (q1,q2 to 0,1) may be possible simultaneously by a bespoke
 * swap routine.
 */
 void statevec_multiControlledTwoQubitUnitary(Qureg qureg, long long int ctrlMask, const int q1, const int q2, ComplexMatrix4 u) {

  //!!simple return in cpu_local 

  int q1FitsInNode = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, q1);
  int q2FitsInNode = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, q2);

  if (q1FitsInNode && q2FitsInNode) {
    statevec_multiControlledTwoQubitUnitaryLocal(qureg, ctrlMask, q1, q2, u);

  } else if (q1FitsInNode) {
    int qSwap = (q1 > 0)? q1-1 : q1+1;
    statevec_swapQubitAmps(qureg, q2, qSwap);
    statevec_multiControlledTwoQubitUnitaryLocal(qureg, ctrlMask, q1, qSwap, u);
    statevec_swapQubitAmps(qureg, q2, qSwap);

  } else if (q2FitsInNode) {
    int qSwap = (q2 > 0)? q2-1 : q2+1;
    statevec_swapQubitAmps(qureg, q1, qSwap);
    statevec_multiControlledTwoQubitUnitaryLocal(qureg, ctrlMask, qSwap, q2, u);
    statevec_swapQubitAmps(qureg, q1, qSwap);

  } else {
    int swap1 = 0;
    int swap2 = 1;
    statevec_swapQubitAmps(qureg, q1, swap1);
    statevec_swapQubitAmps(qureg, q2, swap2);
    statevec_multiControlledTwoQubitUnitaryLocal(qureg, ctrlMask, swap1, swap2, u);
    statevec_swapQubitAmps(qureg, q1, swap1);
    statevec_swapQubitAmps(qureg, q2, swap2);
  }
}

/** This calls swapQubitAmps only when it would involve a distributed communication;
 * if the qubit chunks already fit in the node, it operates the unitary direct.
 * It is already gauranteed here that all target qubits can fit on each node (this is
 * validated in the front-end)
 *
 * @TODO: refactor so that the 'swap back' isn't performed; instead the qubit locations
 * are updated.
 */
 void statevec_multiControlledMultiQubitUnitary(Qureg qureg, long long int ctrlMask, int* targs, const int numTargs, ComplexMatrixN u) {

  //!!simple return in cpu_local
  // only these functions are related to gpu process:
  // statevec_swapQubitAmps()
  // statevec_multiControlledMultiQubitUnitaryLocal()

  // bit mask of target qubits (for quick collision checking)
  long long int targMask = getQubitBitMask(targs, numTargs);

  // find lowest qubit available for swapping (isn't in targs)
  int freeQb=0;
  while (maskContainsBitOnCPU(targMask, freeQb))
      freeQb++;

  // assign indices of where each target will be swapped to (else itself)
  int swapTargs[numTargs];
  for (int t=0; t<numTargs; t++) {
      if (halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, targs[t]))
          swapTargs[t] = targs[t];
      else {
          // mark swap
          swapTargs[t] = freeQb;

          // update ctrlMask if swapped-out qubit was a control
          if (maskContainsBitOnCPU(ctrlMask, swapTargs[t]))
              ctrlMask = flipBitOnCPU(flipBitOnCPU(ctrlMask, swapTargs[t]), targs[t]); // swap targ and ctrl

          // locate next available on-chunk qubit
          freeQb++;
          while (maskContainsBitOnCPU(targMask, freeQb))
              freeQb++;
      }
  }

  // perform swaps as necessary
  for (int t=0; t<numTargs; t++)
      if (swapTargs[t] != targs[t])
          statevec_swapQubitAmpsLocal(qureg, targs[t], swapTargs[t]);

  // all target qubits have now been swapped into local memory
  statevec_multiControlledMultiQubitUnitaryLocal(qureg, ctrlMask, swapTargs, numTargs, u);

  // undo swaps
  for (int t=0; t<numTargs; t++)
      if (swapTargs[t] != targs[t])
          statevec_swapQubitAmpsLocal(qureg, targs[t], swapTargs[t]);
}




// Modified from  QuEST_gpu_local.cu (QuEST_gpu.cu)

void statevec_createQureg(Qureg *qureg, int numQubits, QuESTEnv env)
{   
    // stage 1 done! not very sure.

    // ~~allocate CPU memory~~
    // this part is same with cpu local +yh

    long long int numAmps = 1L << numQubits;
    long long int numAmpsPerRank = numAmps/env.numRanks;
    // fix pointers problems from origin QuEST-kit repo. yh 2021.3.28
    // qureg->stateVec.real = (qreal*) malloc(numAmpsPerRank * sizeof(*(qureg->stateVec.real)));
    // qureg->stateVec.imag = (qreal*) malloc(numAmpsPerRank * sizeof(*(qureg->stateVec.imag)));
    // if (env.numRanks>1){
    //     qureg->pairStateVec.real = (qreal*) malloc(numAmpsPerRank * sizeof(*(qureg->pairStateVec.real)));
    //     qureg->pairStateVec.imag = (qreal*) malloc(numAmpsPerRank * sizeof(*(qureg->pairStateVec.imag)));
    // }

    cudaMalloc(&(qureg->stateVec.real), numAmpsPerRank * sizeof(*(qureg->stateVec.real)));
    cudaMalloc(&(qureg->stateVec.imag), numAmpsPerRank * sizeof(*(qureg->stateVec.imag)));
    if (env.numRanks > 1) {
      cudaMalloc(&(qureg->pairStateVec.real), numAmpsPerRank * sizeof(*(qureg->pairStateVec.real)));
      cudaMalloc(&(qureg->pairStateVec.imag), numAmpsPerRank * sizeof(*(qureg->pairStateVec.imag)));
    }

    // check gpu memory allocation was successful
    if ( (!(qureg->stateVec.real) || !(qureg->stateVec.imag))
            && numAmpsPerRank ) {
        printf("Could not allocate memory!\n");
        exit (EXIT_FAILURE);
    }
    if ( env.numRanks>1 && (!(qureg->pairStateVec.real) || !(qureg->pairStateVec.imag))
            && numAmpsPerRank ) {
        printf("Could not allocate memory!\n");
        exit (EXIT_FAILURE);
    }

    qureg->numQubitsInStateVec = numQubits;
    qureg->numAmpsPerChunk = numAmpsPerRank;
    qureg->numAmpsTotal = numAmps;
    qureg->chunkId = env.rank;
    qureg->numChunks = env.numRanks;
    qureg->isDensityMatrix = 0;

    // allocate GPU memory
    // cudaMalloc(&(qureg->deviceStateVec.real), qureg->numAmpsPerChunk*sizeof(*(qureg->deviceStateVec.real)));
    // cudaMalloc(&(qureg->deviceStateVec.imag), qureg->numAmpsPerChunk*sizeof(*(qureg->deviceStateVec.imag)));
    cudaMalloc(&(qureg->firstLevelReduction), ceil(numAmpsPerRank/(qreal)REDUCE_SHARED_SIZE)*sizeof(qreal));
    cudaMalloc(&(qureg->secondLevelReduction), ceil(numAmpsPerRank/(qreal)(REDUCE_SHARED_SIZE*REDUCE_SHARED_SIZE))*
            sizeof(qreal));

    // check gpu memory allocation was successful
    // if (!(qureg->deviceStateVec.real) || !(qureg->deviceStateVec.imag)){
    //     printf("Could not allocate memory on GPU!\n");
    //     exit (EXIT_FAILURE);
    // }

    // Multiple messages are required as cuMPI uses int rather than long long int for count
    // For openmpi, messages are further restricted to 2GB in size -- do this for all cases
    // to be safe
    long long int maxMessageCount = cuMPI_MAX_AMPS_IN_MSG;
    if (qureg->numAmpsPerChunk < maxMessageCount)
        maxMessageCount = qureg->numAmpsPerChunk;

    // safely assume cuMPI_MAX... = 2^n, so division always exact
    int numMessages = qureg->numAmpsPerChunk/maxMessageCount;

    int i;
    for (i=0; i<numMessages; i++) {
      cuMPI_NewGlobalComm(&pipeReal[i]);
      cuMPI_NewGlobalComm(&pipeImag[i]);
    }


}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env)
{
    // stage 1 done!
    // add extra reset from cpu local.
    qureg.numQubitsInStateVec = 0;
    qureg.numAmpsTotal = 0;
    qureg.numAmpsPerChunk = 0;

    // Free CPU memory
    // free(qureg.stateVec.real);
    // free(qureg.stateVec.imag);
    cudaFree(qureg.stateVec.real);
    cudaFree(qureg.stateVec.imag);
    if (env.numRanks>1){
        // free(qureg.pairStateVec.real);
        // free(qureg.pairStateVec.imag);
        cudaFree(qureg.pairStateVec.real);
        cudaFree(qureg.pairStateVec.imag);
    }
    qureg.stateVec.real = NULL;
    qureg.stateVec.imag = NULL;
    qureg.pairStateVec.real = NULL;
    qureg.pairStateVec.imag = NULL;

    // Free GPU memory
    // cudaFree(qureg.deviceStateVec.real);
    // cudaFree(qureg.deviceStateVec.imag);
    // ? cudaFree(qureg.firstLevelReduction.real);
}

__global__ void statevec_initBlankStateKernel(long long int stateVecSize, qreal *stateVecReal, qreal *stateVecImag){
  long long int index;

  // initialise the statevector to be all-zeros
  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;
  stateVecReal[index] = 0.0;
  stateVecImag[index] = 0.0;
}

void statevec_initBlankState(Qureg qureg)
{
  // stage 1 done!

  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_initBlankStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, 
      qureg.stateVec.real, 
      qureg.stateVec.imag);
}

// especially for qureg.chunkId == 0
__global__ void statevec_initZeroStateKernel(long long int stateVecSize, qreal *stateVecReal, qreal *stateVecImag){
  long long int index;

  // initialise the state to |0000..0000>
  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;
  stateVecReal[index] = 0.0;
  stateVecImag[index] = 0.0;

  if (index==0){
      // zero state |0000..0000> has probability 1
      stateVecReal[0] = 1.0;
      stateVecImag[0] = 0.0;
  }
}

void statevec_initZeroState(Qureg qureg)
{
  // stage 1 done!

  if (qureg.chunkId==0) {

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);

    // zero state |0000..0000> has probability 1
    statevec_initZeroStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, 
      qureg.stateVec.real, 
      qureg.stateVec.imag
    );
  } else {

    statevec_initBlankState(qureg);
  }

}

// returns 1 if successful, else 0
int statevec_initStateFromSingleFile(Qureg *qureg, char filename[200], QuESTEnv env){
  // stage 1 done!

  long long int chunkSize, stateVecSize;
  long long int indexInChunk, totalIndex;

  chunkSize = qureg->numAmpsPerChunk;
  stateVecSize = chunkSize*qureg->numChunks;

  // qreal *stateVecReal = qureg->stateVec.real;
  // qreal *stateVecImag = qureg->stateVec.imag;
  qreal *tempStateVecReal = (qreal *)malloc(chunkSize * sizeof(qreal));
  qreal *tempStateVecImag = (qreal *)malloc(chunkSize * sizeof(qreal));

  FILE *fp;
  char line[200];

  for (int rank=0; rank<(qureg->numChunks); rank++){
      if (rank==qureg->chunkId){
          fp = fopen(filename, "r");

          // indicate file open failure
          if (fp == NULL)
              return 0;

          indexInChunk = 0; totalIndex = 0;
          while (fgets(line, sizeof(char)*200, fp) != NULL && totalIndex<stateVecSize){
              if (line[0]!='#'){
                  int chunkId = (int) (totalIndex/chunkSize);
                  if (chunkId==qureg->chunkId){
                      # if QuEST_PREC==1
                      sscanf(line, "%f, %f", &(tempStateVecReal[indexInChunk]),
                              &(tempStateVecImag[indexInChunk]));
                      # elif QuEST_PREC==2
                      sscanf(line, "%lf, %lf", &(tempStateVecReal[indexInChunk]),
                              &(tempStateVecImag[indexInChunk]));
                      # elif QuEST_PREC==4
                      sscanf(line, "%Lf, %Lf", &(tempStateVecReal[indexInChunk]),
                              &(tempStateVecImag[indexInChunk]));
                      # endif
                      indexInChunk += 1;
                  }
                  totalIndex += 1;
              }
          }
          fclose(fp);
      }
      syncQuESTEnv(env);
  }

  // copy state to GPU
  cudaDeviceSynchronize();
  if (DEBUG) printf("Copying data to GPU\n");
  cudaMemcpy(qureg->stateVec.real, tempStateVecReal, 
          chunkSize*sizeof(qreal), cudaMemcpyHostToDevice);
  cudaMemcpy(qureg->stateVec.imag, tempStateVecImag, 
          chunkSize*sizeof(qreal), cudaMemcpyHostToDevice);
  if (DEBUG) printf("Finished copying data to GPU\n");


  // indicate success
  return 1;
}


void statevec_setAmps(Qureg qureg, long long int startInd, qreal* reals, qreal* imags, long long int numAmps) {
  // stage 1 done!
  
  /* this is actually distributed, since the user's code runs on every node */

  // local start/end indices of the given amplitudes, assuming they fit in this chunk
  // these may be negative or above qureg.numAmpsPerChunk
  long long int localStartInd = startInd - qureg.chunkId*qureg.numAmpsPerChunk;
  long long int localEndInd = localStartInd + numAmps; // exclusive

  // add this to a local index to get corresponding elem in reals & imags
  long long int offset = qureg.chunkId*qureg.numAmpsPerChunk - startInd;

  // restrict these indices to fit into this chunk
  if (localStartInd < 0)
      localStartInd = 0;
  if (localEndInd > qureg.numAmpsPerChunk)
      localEndInd = qureg.numAmpsPerChunk;
  // they may now be out of order = no iterations

  // unpacking OpenMP vars
  // long long int index;
  qreal* vecRe = qureg.stateVec.real;
  qreal* vecIm = qureg.stateVec.imag;

  // iterate these local inds - this might involve no iterations
  if (localStartInd < localEndInd) {
      size_t copyCount = (size_t)((localEndInd - localStartInd) * sizeof(*(qureg.stateVec.real)));
      cudaDeviceSynchronize();
      cudaMemcpy(
          vecRe + localStartInd,
          reals + localStartInd + offset,
          copyCount,
          cudaMemcpyHostToDevice);
      cudaMemcpy(
          vecIm + localStartInd,
          imags + localStartInd + offset,
          copyCount,
          cudaMemcpyHostToDevice);
  }
  // for (index=localStartInd; index < localEndInd; index++) {
  //     vecRe[index] = reals[index + offset];
  //     vecIm[index] = imags[index + offset];
  // }

  // Old single GPU version:
  // cudaDeviceSynchronize();
  // cudaMemcpy(
  //     qureg.deviceStateVec.real + startInd, 
  //     reals,
  //     numAmps * sizeof(*(qureg.deviceStateVec.real)), 
  //     cudaMemcpyHostToDevice);
  // cudaMemcpy(
  //     qureg.deviceStateVec.imag + startInd,
  //     imags,
  //     numAmps * sizeof(*(qureg.deviceStateVec.real)), 
  //     cudaMemcpyHostToDevice);
}

/** works for both statevectors and density matrices */
void statevec_cloneQureg(Qureg targetQureg, Qureg copyQureg) {
  // stage 1 done!

  // copy copyQureg's GPU statevec to targetQureg's GPU statevec
  cudaDeviceSynchronize();
  cudaMemcpy(
      targetQureg.stateVec.real, 
      copyQureg.stateVec.real, 
      targetQureg.numAmpsPerChunk*sizeof(*(targetQureg.stateVec.real)), 
      cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      targetQureg.stateVec.imag, 
      copyQureg.stateVec.imag, 
      targetQureg.numAmpsPerChunk*sizeof(*(targetQureg.stateVec.imag)), 
      cudaMemcpyDeviceToDevice);
}

void getEnvironmentString(QuESTEnv env, Qureg qureg, char str[200]){
  sprintf(str, "%dqubits_GPU_noMpi_noOMP", qureg.numQubitsInStateVec);    
}

void copyStateToGPU(Qureg qureg)
{
  assert( 0 );
  // don't call this function, the content has been injected into certain code.
}

void copyStateFromGPU(Qureg qureg)
{
  assert( 0 );
  // don't call this function, the content has been injected into certain code.
}

/** Print the current state vector of probability amplitudes for a set of qubits to standard out. 
  For debugging purposes. Each rank should print output serially. Only print output for systems <= 5 qubits
 */
__global__ void statevec_reportStateToScreenSingleKernel(
  const long long int chunkSize, 
  qreal *stateVecReal, 
  qreal *stateVecImag
){
  long long int index;
  for(index=0; index<chunkSize; index++){
      //printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n", qureg.pairStateVec.real[index], qureg.pairStateVec.imag[index]);
      printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n", stateVecReal[index], stateVecImag[index]);
  }
}
void statevec_reportStateToScreen(Qureg qureg, QuESTEnv env, int reportRank){
  // stage 1 done!
  
  int rank;
  if (qureg.numQubitsInStateVec<=5){
      for (rank=0; rank<qureg.numChunks; rank++){
          if (qureg.chunkId==rank){
              if (reportRank) {
                  printf("Reporting state from rank %d [\n", qureg.chunkId);
                  printf("real, imag\n");
              } else if (rank==0) {
                  printf("Reporting state [\n");
                  printf("real, imag\n");
              }

              cudaDeviceSynchronize();
              statevec_reportStateToScreenSingleKernel<<<1, 1>>> (qureg.numAmpsPerChunk, qureg.stateVec.real, qureg.stateVec.imag);
              
              if (reportRank || rank==qureg.numChunks-1) printf("]\n");
          }
          syncQuESTEnv(env);
      }
  } else printf("Error: reportStateToScreen will not print output for systems of more than 5 qubits.\n");
}

__global__ void statevec_initPlusStateKernel(
  long long int chunkSize, 
  qreal *stateVecReal, 
  qreal *stateVecImag, 
  qreal normFactor
){
  long long int index;

  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=chunkSize) return;

  // qreal normFactor = 1.0/sqrt((qreal)stateVecSize);
  stateVecReal[index] = normFactor;
  stateVecImag[index] = 0.0;
}

void statevec_initPlusState(Qureg qureg)
{
  // stage 1 done!
  
  long long int chunkSize, stateVecSize;
  // long long int index;

  // dimension of the state vector
  chunkSize = qureg.numAmpsPerChunk;
  stateVecSize = chunkSize*qureg.numChunks;
  qreal normFactor = 1.0/sqrt((qreal)stateVecSize);

  // initialise the state to |+++..+++> = 1/normFactor {1, 1, 1, ...}
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(chunkSize)/threadsPerCUDABlock);
  statevec_initPlusStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      chunkSize, 
      qureg.stateVec.real, 
      qureg.stateVec.imag, 
      normFactor);
}

__global__ void statevec_initClassicalStateKernel(
  long long int stateVecSize, 
  qreal *stateVecReal, 
  qreal *stateVecImag, 
  long long int stateInd
){
  long long int index;

  // initialise the state to |stateInd>
  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;
  stateVecReal[index] = 0.0;
  stateVecImag[index] = 0.0;

  if (index==stateInd){
      // classical state has probability 1
      stateVecReal[stateInd] = 1.0;
      stateVecImag[stateInd] = 0.0;
  }
}

void statevec_initClassicalState(Qureg qureg, long long int stateInd)
{
  // stage 1 done!
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);

  // dimension of the state vector
  long long int stateVecSize = qureg.numAmpsPerChunk;

  // give the specified classical state prob 1
  if (qureg.chunkId == stateInd/stateVecSize){
      statevec_initClassicalStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
          stateVecSize, 
          qureg.stateVec.real, 
          qureg.stateVec.imag,
          stateInd % stateVecSize
      );
  } else {
      statevec_initClassicalStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
          stateVecSize, 
          qureg.stateVec.real, 
          qureg.stateVec.imag,
          stateVecSize // chunkId not match, so index==stateInd(=stateVecSize) will always be 0
      );
  }
}


#ifdef __cplusplus
}
#endif
