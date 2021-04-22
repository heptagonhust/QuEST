#include "QuEST_gpu_local.cu"
#include "QuEST_gpu.h"
#include "cuMPI/cuMPI_runtime.h"
#include "cuMPI/src/cuMPI_runtime.h"

/********************** For cuMPI environment **********************/
int myRank;                 // cuMPI comm local ranks
int nRanks;                 // total cuMPI comm ranks
int localRank;              // CUDA device ID

ncclUniqueId id;            // NCCL Unique ID
cuMPI_Comm comm;            // cuMPI comm
cudaStream_t defaultStream; // CUDA stream generated for each GPU
uint64_t hostHashs[10];     // host name hash in cuMPI
char hostname[1024];        // host name for identification in cuMPI
/*******************************************************************/

//cudampiinit
//cudaAllreduce
//cuMPI_Broadcast
//cuMPI_SendRecv

void copyStateFromCurrentGPU(Qureg qureg){
  copyStateFromGPU(qureg);
}

Complex statevec_calcInnerProduct(Qureg bra, Qureg ket) {
  // stage 1 done! (mode 1)

  Complex localInnerProd = statevec_calcInnerProductLocal(bra, ket);
  if (bra.numChunks == 1)
    return localInnerProd;
  
  qreal localReal = localInnerProd.real;
  qreal localImag = localInnerProd.imag;
  qreal globalReal, globalImag;
  cuMPI_Allreduce(&localReal, &globalReal, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);
  cuMPI_Allreduce(&localImag, &globalImag, 1, cuMPI_QuEST_REAL, cuMPI_SUM, cuMPI_COMM_WORLD);

  Complex globalInnerProd;
  globalInnerProd.real = globalReal;
  globalInnerProd.imag = globalImag;
  return globalInnerProd;
}

qreal densmatr_calcTotalProb(Qureg qureg) {
  // phase 1 done! (mode 2)
  // gpu local is almost same with cpu local

	// computes the trace by summing every element ("diag") with global index (2^n + 1)i for i in [0, 2^n-1]

	// computes first local index containing a diagonal element
	long long int diagSpacing = 1LL + (1LL << qureg.numQubitsRepresented);

  copyStateFromCurrentGPU(qureg);

  long long int numPrevDiags = (qureg.chunkId>0)? 1+(qureg.chunkId*qureg.numAmpsPerChunk)/diagSpacing : 0;
  long long int globalIndNextDiag = diagSpacing * numPrevDiags;
  long long int localIndNextDiag = globalIndNextDiag % qureg.numAmpsPerChunk;
  long long int index;

  qreal rankTotal = 0;
  qreal y, t, c;
  c = 0;

  // iterates every local diagonal
  for (index=localIndNextDiag; index < qureg.numAmpsPerChunk; index += diagSpacing) {

    // Kahan summation - brackets are important
    y = qureg.stateVec.real[index] - c;
    t = rankTotal + y;
    c = ( t - rankTotal ) - y;
    rankTotal = t;
  }

  // combine each node's sum of diagonals
  qreal globalTotal;
  if (qureg.numChunks > 1)
    MPI_Allreduce(&rankTotal, &globalTotal, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
  else
    globalTotal = rankTotal;

  return globalTotal;
}

qreal statevec_calcTotalProbLocal(Qureg qureg){
  // phase 1 done! (mode 2)
  // gpu local is almost same with cpu local

  // Implemented using Kahan summation for greater accuracy at a slight floating
  //   point operation overhead. For more details see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
  qreal pTotal=0;
  qreal y, t, c;
  qreal allRankTotals=0;
  long long int index;
  long long int numAmpsPerRank = qureg.numAmpsPerChunk;

  copyStateFromCurrentGPU(qureg);

  c = 0.0;
  for (index=0; index<numAmpsPerRank; index++){
      // Perform pTotal+=qureg.stateVec.real[index]*qureg.stateVec.real[index]; by Kahan
      y = qureg.stateVec.real[index]*qureg.stateVec.real[index] - c;
      t = pTotal + y;
      // Don't change the bracketing on the following line
      c = ( t - pTotal ) - y;
      pTotal = t;
      // Perform pTotal+=qureg.stateVec.imag[index]*qureg.stateVec.imag[index]; by Kahan
      y = qureg.stateVec.imag[index]*qureg.stateVec.imag[index] - c;
      t = pTotal + y;
      // Don't change the bracketing on the following line
      c = ( t - pTotal ) - y;
      pTotal = t;
  }
  if (qureg.numChunks>1)
    MPI_Allreduce(&pTotal, &allRankTotals, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
  else
    allRankTotals=pTotal;

  return allRankTotals;
}

static int isChunkToSkipInFindPZero(int chunkId, long long int chunkSize, int measureQubit);
static int chunkIsUpper(int chunkId, long long int chunkSize, int targetQubit);
static int chunkIsUpperInOuterBlock(int chunkId, long long int chunkSize, int targetQubit, int numQubits);
static void getRotAngle(int chunkIsUpper, Complex *rot1, Complex *rot2, Complex alpha, Complex beta);
static int getChunkPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit);
static int getChunkOuterBlockPairId(int chunkIsUpper, int chunkId, long long int chunkSize, int targetQubit, int numQubits);
static int halfMatrixBlockFitsInChunk(long long int chunkSize, int targetQubit);
static int getChunkIdFromIndex(Qureg qureg, long long int index);

int getChunkIdFromIndex(Qureg qureg, long long int index){
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
  // stage 1 done! changed to cuMPI.
  
  // Local version is similar to cpu_local version. +yh

  if (!GPUExists()){
    printf("Trying to run GPU code with no GPU available\n");
    exit(EXIT_FAILURE);
  }

  QuESTEnv env;

  // init MPI environment
  // int rank, numRanks, initialized;
  int initialized;
  cuMPI_Initialized(&initialized);
  if (!initialized){

    cuMPI_Init(NULL, NULL);
    // cuMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    // cuMPI_Comm_rank(MPI_COMM_WORLD, &rank);

    env.rank=myRank;
    env.numRanks=nRanks;
    // cuAllocateOneGPUPerProcess();

  } else {

    printf("ERROR: Trying to initialize QuESTEnv multiple times. Ignoring...\n");

    // ensure env is initialised anyway, so the compiler is happy
    // MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    env.rank=rank;
    env.numRanks=numRanks;

  }

  seedQuESTDefault();

  return env;
}

void syncQuESTEnv(QuESTEnv env){
  // stage 1 done!
  // After computation in GPU device is done, synchronize MPI message. 
  cudaDeviceSynchronize();
  cuMPI_Barrier(cuMPI_COMM_WORLD);
}

int syncQuESTSuccess(int successCode){
  // stage 1 done!
  // nothing to do for GPU method.
  int totalSuccess;
  // MPI_LAND logic and
  cuMPI_Allreduce(&successCode, &totalSuccess, 1, cuMPI_INT, cuMPI_MIN, cuMPI_COMM_WORLD);
  return totalSuccess;
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
  // phase 1 done! (mode 3)
  // direct copy from device state memory

  int chunkId = getChunkIdFromIndex(qureg, index);
  qreal el=0;
  if (qureg.chunkId==chunkId){
      // el = qureg.stateVec.real[index-chunkId*qureg.numAmpsPerChunk];
      cudaMemcpy(&el, &(qureg.deviceStateVec.real[index-chunkId*qureg.numAmpsPerChunk]), 
        sizeof(*(qureg.deviceStateVec.real)), cudaMemcpyDeviceToHost);
  }
  cuMPI_Bcast(&el, 1, MPI_QuEST_REAL, chunkId, MPI_COMM_WORLD);
  return el;
}

qreal statevec_getImagAmp(Qureg qureg, long long int index){
  // phase 1 done! (mode 3)
  // direct copy from device state memory

  int chunkId = getChunkIdFromIndex(qureg, index);
  qreal el=0;
  if (qureg.chunkId==chunkId){
      //el = qureg.stateVec.imag[index-chunkId*qureg.numAmpsPerChunk];
      cudaMemcpy(&el, &(qureg.deviceStateVec.imag[index-chunkId*qureg.numAmpsPerChunk]), 
        sizeof(*(qureg.deviceStateVec.imag)), cudaMemcpyDeviceToHost);
  }
  cuMPI_Bcast(&el, 1, MPI_QuEST_REAL, chunkId, MPI_COMM_WORLD);
  return el;
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

/** This copies/clones vec (a statevector) into every node's matr pairState.
 * (where matr is a density matrix or equal number of qubits as vec) */
void copyVecIntoMatrixPairState(Qureg matr, Qureg vec) {

    // Remember that for every amplitude that `vec` stores on the node,
    // `matr` stores an entire column. Ergo there are always an integer
    // number (in fact, a power of 2) number of  `matr`s columns on each node.
    // Since the total size of `vec` (between all nodes) is one column
    // and each node stores (possibly) multiple columns (vec.numAmpsPerChunk as many),
    // `vec` can be fit entirely inside a single node's matr.pairStateVec (with excess!)

    // copy this node's vec segment into this node's matr pairState (in the right spot)
    long long int numLocalAmps = vec.numAmpsPerChunk;
    long long int myOffset = vec.chunkId * numLocalAmps;
    memcpy(&matr.pairStateVec.real[myOffset], vec.stateVec.real, numLocalAmps * sizeof(qreal));
    memcpy(&matr.pairStateVec.imag[myOffset], vec.stateVec.imag, numLocalAmps * sizeof(qreal));

    // we now want to share this node's vec segment with other node, so that
    // vec is cloned in every node's matr.pairStateVec

    // work out how many messages needed to send vec chunks (2GB limit)
    long long int maxMsgSize = MPI_MAX_AMPS_IN_MSG;
    if (numLocalAmps < maxMsgSize)
        maxMsgSize = numLocalAmps;
    // safely assume MPI_MAX... = 2^n, so division always exact:
    int numMsgs = numLocalAmps / maxMsgSize;

    // every node gets a turn at being the broadcaster
    for (int broadcaster=0; broadcaster < vec.numChunks; broadcaster++) {

        long long int otherOffset = broadcaster * numLocalAmps;

        // every node sends a slice of qureg's pairState to every other
        for (int i=0; i< numMsgs; i++) {

            // by sending that slice in further slices (due to bandwidth limit)
            MPI_Bcast(
                &matr.pairStateVec.real[otherOffset + i*maxMsgSize],
                maxMsgSize,  MPI_QuEST_REAL, broadcaster, MPI_COMM_WORLD);
            MPI_Bcast(
                &matr.pairStateVec.imag[otherOffset + i*maxMsgSize],
                maxMsgSize,  MPI_QuEST_REAL, broadcaster, MPI_COMM_WORLD);
        }
    }
}

/********************************************/

qreal densmatr_calcFidelity(Qureg qureg, Qureg pureState) {
  // phase 1 undone!!!!!!!!
  // concern about cpu_local version +yh
  // !!copyVecIntoMatrixPairState

  // set qureg's pairState is to be the full pureState (on every node)
  // this function call is same with cpu local like below: +yh
  // 1. save pointers to qureg's pair state
  // 2. populate qureg pair state with pure state (by repointing)
  // 3. restore pointers
  copyVecIntoMatrixPairState(qureg, pureState);

  // collect calcFidelityLocal by every machine
  qreal localSum = densmatr_calcFidelityLocal(qureg, pureState);

  // sum each localSum
  qreal globalSum;
  MPI_Allreduce(&localSum, &globalSum, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);

  return globalSum;
}

qreal densmatr_calcHilbertSchmidtDistance(Qureg a, Qureg b) {
  // phase 1 done! (mode1)
  // gpu local function was modified.
  qreal localSum = densmatr_calcHilbertSchmidtDistanceSquaredLocal(a, b);

  qreal globalSum;
  MPI_Allreduce(&localSum, &globalSum, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);

  qreal dist = sqrt(globalSum);
  return dist;
}

qreal densmatr_calcInnerProduct(Qureg a, Qureg b) {
  // phase 1 done! (mode1)
  // cpu local wrapper function just return local call.
  qreal localSum = densmatr_calcInnerProductLocal(a, b);

  qreal globalSum;
  MPI_Allreduce(&localSum, &globalSum, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);

  qreal dist = globalSum;
  return dist;
}

void densmatr_initPureState(Qureg targetQureg, Qureg copyQureg) {
  // phase 1 undone!!!!!!!!
  // similar to densmatr_calcFidelity.
  // concern about cpu_local version +yh
  // !!copyVecIntoMatrixPairState

  if (targetQureg.numChunks==1){
      // local version
      // save pointers to qureg's pair state
      qreal* quregPairRePtr = targetQureg.pairStateVec.real;
      qreal* quregPairImPtr = targetQureg.pairStateVec.imag;

      // populate qureg pair state with pure state (by repointing)
      targetQureg.pairStateVec.real = copyQureg.stateVec.real;
      targetQureg.pairStateVec.imag = copyQureg.stateVec.imag;

      // populate density matrix via it's pairState
      densmatr_initPureStateLocal(targetQureg, copyQureg);

      // restore pointers
      targetQureg.pairStateVec.real = quregPairRePtr;
      targetQureg.pairStateVec.imag = quregPairImPtr;
  } else {
      // set qureg's pairState is to be the full pure state (on every node)
      copyVecIntoMatrixPairState(targetQureg, copyQureg);

      // update every density matrix chunk using pairState
      densmatr_initPureStateLocal(targetQureg, copyQureg);
  }
}






/************** copy from distributed cpu version **************/






void exchangeStateVectors(Qureg qureg, int pairRank){
  // stage 1 done!

  // MPI send/receive vars
  int TAG=100;
  MPI_Status status;

  // Multiple messages are required as MPI uses int rather than long long int for count
  // For openmpi, messages are further restricted to 2GB in size -- do this for all cases
  // to be safe
  long long int maxMessageCount = MPI_MAX_AMPS_IN_MSG;
  if (qureg.numAmpsPerChunk < maxMessageCount)
      maxMessageCount = qureg.numAmpsPerChunk;

  // safely assume MPI_MAX... = 2^n, so division always exact
  int numMessages = qureg.numAmpsPerChunk/maxMessageCount;
  int i;
  long long int offset;
  // send my state vector to pairRank's qureg.pairStateVec
  // receive pairRank's state vector into qureg.pairStateVec
  for (i=0; i<numMessages; i++){
      offset = i*maxMessageCount;
      cuMPI_Sendrecv(&qureg.stateVec.real[offset], maxMessageCount, MPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.real[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &status);
      //printf("rank: %d err: %d\n", qureg.rank, err);
      cuMPI_Sendrecv(&qureg.stateVec.imag[offset], maxMessageCount, MPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.imag[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &status);
  }
}

void exchangePairStateVectorHalves(Qureg qureg, int pairRank){
  // MPI send/receive vars
  int TAG=100;
  MPI_Status status;
  long long int numAmpsToSend = qureg.numAmpsPerChunk >> 1;

  // Multiple messages are required as MPI uses int rather than long long int for count
  // For openmpi, messages are further restricted to 2GB in size -- do this for all cases
  // to be safe
  long long int maxMessageCount = MPI_MAX_AMPS_IN_MSG;
  if (numAmpsToSend < maxMessageCount)
      maxMessageCount = numAmpsToSend;

  // safely assume MPI_MAX... = 2^n, so division always exact
  int numMessages = numAmpsToSend/maxMessageCount;
  int i;
  long long int offset;
  // send the bottom half of my state vector to the top half of pairRank's qureg.pairStateVec
  // receive pairRank's state vector into the top of qureg.pairStateVec
  for (i=0; i<numMessages; i++){
      offset = i*maxMessageCount;
      MPI_Sendrecv(&qureg.pairStateVec.real[offset+numAmpsToSend], maxMessageCount,
              MPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.real[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &status);
      //printf("rank: %d err: %d\n", qureg.rank, err);
      MPI_Sendrecv(&qureg.pairStateVec.imag[offset+numAmpsToSend], maxMessageCount,
              MPI_QuEST_REAL, pairRank, TAG,
              &qureg.pairStateVec.imag[offset], maxMessageCount, MPI_QuEST_REAL,
              pairRank, TAG, MPI_COMM_WORLD, &status);
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

void compressPairVectorForTwoQubitDepolarise(Qureg qureg, const int targetQubit,
      const int qubit2) {

  long long int sizeInnerBlockQ1, sizeInnerHalfBlockQ1;
  long long int sizeInnerBlockQ2, sizeInnerHalfBlockQ2, sizeInnerQuarterBlockQ2;
  long long int sizeOuterColumn, sizeOuterQuarterColumn;
  long long int
       thisInnerBlockQ2,
       thisOuterColumn, // current column in density matrix
       thisIndex,    // current index in (density matrix representation) state vector
       thisIndexInOuterColumn,
       thisIndexInInnerBlockQ1,
       thisIndexInInnerBlockQ2,
       thisInnerBlockQ1InInnerBlockQ2;
  int outerBitQ1, outerBitQ2;

  long long int thisTask;
  const long long int numTasks=qureg.numAmpsPerChunk>>2;

  // set dimensions
  sizeInnerHalfBlockQ1 = 1LL << targetQubit;
  sizeInnerHalfBlockQ2 = 1LL << qubit2;
  sizeInnerQuarterBlockQ2 = sizeInnerHalfBlockQ2 >> 1;
  sizeInnerBlockQ2 = sizeInnerHalfBlockQ2 << 1;
  sizeInnerBlockQ1 = 2LL * sizeInnerHalfBlockQ1;
  sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
  sizeOuterQuarterColumn = sizeOuterColumn >> 2;

# ifdef _OPENMP
# pragma omp parallel \
  shared   (sizeInnerBlockQ1,sizeInnerHalfBlockQ1,sizeInnerQuarterBlockQ2,sizeInnerHalfBlockQ2,sizeInnerBlockQ2, \
              sizeOuterColumn, \
              sizeOuterQuarterColumn,qureg) \
  private  (thisTask,thisInnerBlockQ2,thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
              thisIndexInInnerBlockQ1,thisIndexInInnerBlockQ2,thisInnerBlockQ1InInnerBlockQ2,outerBitQ1,outerBitQ2)
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
          thisOuterColumn = thisTask / sizeOuterQuarterColumn;
          // thisTask % sizeOuterQuarterColumn
          thisIndexInOuterColumn = thisTask&(sizeOuterQuarterColumn-1);
          thisInnerBlockQ2 = thisIndexInOuterColumn / sizeInnerQuarterBlockQ2;
          // thisTask % sizeInnerQuarterBlockQ2;
          thisIndexInInnerBlockQ2 = thisTask&(sizeInnerQuarterBlockQ2-1);
          thisInnerBlockQ1InInnerBlockQ2 = thisIndexInInnerBlockQ2 / sizeInnerHalfBlockQ1;
          // thisTask % sizeInnerHalfBlockQ1;
          thisIndexInInnerBlockQ1 = thisTask&(sizeInnerHalfBlockQ1-1);

          // get index in state vector corresponding to upper inner block
          thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlockQ2*sizeInnerBlockQ2
              + thisInnerBlockQ1InInnerBlockQ2*sizeInnerBlockQ1 + thisIndexInInnerBlockQ1;

          // check if we are in the upper or lower half of an outer block for Q1
          outerBitQ1 = extractBitOnCPU(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
          // if we are in the lower half of an outer block, shift to be in the lower half
          // of the inner block as well (we want to dephase |0><0| and |1><1| only)
          thisIndex += outerBitQ1*(sizeInnerHalfBlockQ1);

          // check if we are in the upper or lower half of an outer block for Q2
          outerBitQ2 = extractBitOnCPU(qubit2, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
          // if we are in the lower half of an outer block, shift to be in the lower half
          // of the inner block as well (we want to dephase |0><0| and |1><1| only)
          thisIndex += outerBitQ2*(sizeInnerQuarterBlockQ2<<1);

          // NOTE: at this point thisIndex should be the index of the element we want to
          // dephase in the chunk of the state vector on this process, in the
          // density matrix representation.
          // thisTask is the index of the pair element in pairStateVec

          // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
          //      + pair[thisTask])/2
          qureg.pairStateVec.real[thisTask+numTasks*2] = qureg.stateVec.real[thisIndex];
          qureg.pairStateVec.imag[thisTask+numTasks*2] = qureg.stateVec.imag[thisIndex];
      }
  }
}






/***************** copy from QuEST_cpu.c *****************/







void densmatr_mixDepolarisingDistributed(Qureg qureg, const int targetQubit, qreal depolLevel) {

  // first do dephase part.
  // TODO -- this might be more efficient to do at the same time as the depolarise if we move to
  // iterating over all elements in the state vector for the purpose of vectorisation
  // TODO -- if we keep this split, move this function to densmatr_mixDepolarising()
  densmatr_mixDephasing(qureg, targetQubit, depolLevel);

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
  sizeInnerBlock = 2LL * sizeInnerHalfBlock;
  sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
  sizeOuterHalfColumn = sizeOuterColumn >> 1;

# ifdef _OPENMP
# pragma omp parallel \
  shared   (sizeInnerBlock,sizeInnerHalfBlock,sizeOuterColumn,sizeOuterHalfColumn,qureg,depolLevel) \
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


          // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
          //      + pair[thisTask])/2
          qureg.stateVec.real[thisIndex] = (1-depolLevel)*qureg.stateVec.real[thisIndex] +
                  depolLevel*(qureg.stateVec.real[thisIndex] + qureg.pairStateVec.real[thisTask])/2;

          qureg.stateVec.imag[thisIndex] = (1-depolLevel)*qureg.stateVec.imag[thisIndex] +
                  depolLevel*(qureg.stateVec.imag[thisIndex] + qureg.pairStateVec.imag[thisTask])/2;
      }
  }
}

void densmatr_mixDampingDistributed(Qureg qureg, const int targetQubit, qreal damping) {
  qreal retain=1-damping;
  qreal dephase=sqrt(1-damping);
  // first do dephase part.
  // TODO -- this might be more efficient to do at the same time as the depolarise if we move to
  // iterating over all elements in the state vector for the purpose of vectorisation
  // TODO -- if we keep this split, move this function to densmatr_mixDepolarising()
  densmatr_mixDampingDephase(qureg, targetQubit, dephase);

  long long int sizeInnerBlock, sizeInnerHalfBlock;
  long long int sizeOuterColumn, sizeOuterHalfColumn;
  long long int thisInnerBlock, // current block
       thisOuterColumn, // current column in density matrix
       thisIndex,    // current index in (density matrix representation) state vector
       thisIndexInOuterColumn,
       thisIndexInInnerBlock;
  int outerBit;
  int stateBit;

  long long int thisTask;
  const long long int numTasks=qureg.numAmpsPerChunk>>1;

  // set dimensions
  sizeInnerHalfBlock = 1LL << targetQubit;
  sizeInnerBlock = 2LL * sizeInnerHalfBlock;
  sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
  sizeOuterHalfColumn = sizeOuterColumn >> 1;

# ifdef _OPENMP
# pragma omp parallel \
  shared   (sizeInnerBlock,sizeInnerHalfBlock,sizeOuterColumn,sizeOuterHalfColumn,qureg,damping, retain, dephase) \
  private  (thisTask,thisInnerBlock,thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
              thisIndexInInnerBlock,outerBit, stateBit)
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

          // Extract state bit, is 0 if thisIndex corresponds to a state with 0 in the target qubit
          // and is 1 if thisIndex corresponds to a state with 1 in the target qubit
          stateBit = extractBitOnCPU(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId));

          // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
          //      + pair[thisTask])/2
          if(stateBit == 0){
              qureg.stateVec.real[thisIndex] = qureg.stateVec.real[thisIndex] +
                  damping*( qureg.pairStateVec.real[thisTask]);

              qureg.stateVec.imag[thisIndex] = qureg.stateVec.imag[thisIndex] +
                  damping*( qureg.pairStateVec.imag[thisTask]);
          } else{
              qureg.stateVec.real[thisIndex] = retain*qureg.stateVec.real[thisIndex];

              qureg.stateVec.imag[thisIndex] = retain*qureg.stateVec.imag[thisIndex];
          }
      }
  }
}





/***************************************************************/





void densmatr_mixDepolarising(Qureg qureg, const int targetQubit, qreal depolLevel) {
  // !!need compare to distributed cpu version
  // !!local cpu version is below:
  // if (depolLevel == 0)
  //       return;

  //   densmatr_mixDepolarisingLocal(qureg, targetQubit, depolLevel);

  if (depolLevel == 0)
      return;

  int rankIsUpper; // rank is in the upper half of an outer block
  int pairRank; // rank of corresponding chunk

  int useLocalDataOnly = densityMatrixBlockFitsInChunk(qureg.numAmpsPerChunk,
          qureg.numQubitsRepresented, targetQubit);

  if (useLocalDataOnly){
      densmatr_mixDepolarisingLocal(qureg, targetQubit, depolLevel);
  } else {
      // pack data to send to my pair process into the first half of pairStateVec
      compressPairVectorForSingleQubitDepolarise(qureg, targetQubit);

      rankIsUpper = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit,
              qureg.numQubitsRepresented);
      pairRank = getChunkOuterBlockPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk,
              targetQubit, qureg.numQubitsRepresented);

      exchangePairStateVectorHalves(qureg, pairRank);
      densmatr_mixDepolarisingDistributed(qureg, targetQubit, depolLevel);
  }

}

void densmatr_mixDamping(Qureg qureg, const int targetQubit, qreal damping) {
  // !!need compare to distributed cpu version
  // !!local cpu version is below:
  // if (damping == 0)
  // return;
  // densmatr_mixDampingLocal(qureg, targetQubit, damping);

  if (damping == 0)
      return;

  int rankIsUpper; // rank is in the upper half of an outer block
  int pairRank; // rank of corresponding chunk

  int useLocalDataOnly = densityMatrixBlockFitsInChunk(qureg.numAmpsPerChunk,
          qureg.numQubitsRepresented, targetQubit);

  if (useLocalDataOnly){
      densmatr_mixDampingLocal(qureg, targetQubit, damping);
  } else {
      // pack data to send to my pair process into the first half of pairStateVec
      compressPairVectorForSingleQubitDepolarise(qureg, targetQubit);

      rankIsUpper = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, targetQubit,
              qureg.numQubitsRepresented);
      pairRank = getChunkOuterBlockPairId(rankIsUpper, qureg.chunkId, qureg.numAmpsPerChunk,
              targetQubit, qureg.numQubitsRepresented);

      exchangePairStateVectorHalves(qureg, pairRank);
      densmatr_mixDampingDistributed(qureg, targetQubit, damping);
  }

}

void densmatr_mixTwoQubitDepolarising(Qureg qureg, int qubit1, int qubit2, qreal depolLevel){
  // !!need compare to distributed cpu version
  // !!concern about local cpu version
  
  if (depolLevel == 0)
      return;
  int rankIsUpperBiggerQubit, rankIsUpperSmallerQubit;
  int pairRank; // rank of corresponding chunk
  int biggerQubit, smallerQubit;

  densmatr_mixTwoQubitDephasing(qureg, qubit1, qubit2, depolLevel);

  qreal eta = 2/depolLevel;
  qreal delta = eta - 1 - sqrt( (eta-1)*(eta-1) - 1 );
  qreal gamma = 1+delta;
  gamma = 1/(gamma*gamma*gamma);
  const qreal GAMMA_PARTS_1_OR_2 = 1.0;
  // TODO -- test delta too small
  /*
  if (fabs(4*delta*(1+delta)*gamma-depolLevel)>1e-5){
      printf("Numerical error in delta; for small error rates try Taylor expansion.\n");
      exit(1);
  }
  */

  biggerQubit = qubit1 > qubit2 ? qubit1 : qubit2;
  smallerQubit = qubit1 < qubit2 ? qubit1 : qubit2;
  int useLocalDataOnlyBigQubit, useLocalDataOnlySmallQubit;

  useLocalDataOnlyBigQubit = densityMatrixBlockFitsInChunk(qureg.numAmpsPerChunk,
      qureg.numQubitsRepresented, biggerQubit);
  if (useLocalDataOnlyBigQubit){
      // does parts 1, 2 and 3 locally in one go
      densmatr_mixTwoQubitDepolarisingLocal(qureg, qubit1, qubit2, delta, gamma);
  } else {
      useLocalDataOnlySmallQubit = densityMatrixBlockFitsInChunk(qureg.numAmpsPerChunk,
          qureg.numQubitsRepresented, smallerQubit);
      if (useLocalDataOnlySmallQubit){
          // do part 1 locally
          densmatr_mixTwoQubitDepolarisingLocalPart1(qureg, smallerQubit, biggerQubit, delta);

          // do parts 2 and 3 distributed (if part 2 is distributed part 3 is also distributed)
          // part 2 will be distributed and the value of the small qubit won't matter
          compressPairVectorForTwoQubitDepolarise(qureg, smallerQubit, biggerQubit);
          rankIsUpperBiggerQubit = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, biggerQubit,
                  qureg.numQubitsRepresented);
          pairRank = getChunkOuterBlockPairId(rankIsUpperBiggerQubit, qureg.chunkId, qureg.numAmpsPerChunk,
                  biggerQubit, qureg.numQubitsRepresented);

          exchangePairStateVectorHalves(qureg, pairRank);
          densmatr_mixTwoQubitDepolarisingDistributed(qureg, smallerQubit, biggerQubit, delta, GAMMA_PARTS_1_OR_2);

          // part 3 will be distributed but involve rearranging for the smaller qubit
          compressPairVectorForTwoQubitDepolarise(qureg, smallerQubit, biggerQubit);
          rankIsUpperBiggerQubit = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, biggerQubit,
                  qureg.numQubitsRepresented);
          pairRank = getChunkOuterBlockPairId(rankIsUpperBiggerQubit, qureg.chunkId, qureg.numAmpsPerChunk,
                  biggerQubit, qureg.numQubitsRepresented);

          exchangePairStateVectorHalves(qureg, pairRank);
          densmatr_mixTwoQubitDepolarisingQ1LocalQ2DistributedPart3(qureg, smallerQubit, biggerQubit, delta, gamma);
      } else {
          // do part 1, 2 and 3 distributed
          // part 1
          compressPairVectorForTwoQubitDepolarise(qureg, smallerQubit, biggerQubit);
          rankIsUpperSmallerQubit = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, smallerQubit,
                  qureg.numQubitsRepresented);
          pairRank = getChunkOuterBlockPairId(rankIsUpperSmallerQubit, qureg.chunkId, qureg.numAmpsPerChunk,
                  smallerQubit, qureg.numQubitsRepresented);

          exchangePairStateVectorHalves(qureg, pairRank);
          densmatr_mixTwoQubitDepolarisingDistributed(qureg, smallerQubit, biggerQubit, delta, GAMMA_PARTS_1_OR_2);

          // part 2
          compressPairVectorForTwoQubitDepolarise(qureg, smallerQubit, biggerQubit);
          rankIsUpperBiggerQubit = chunkIsUpperInOuterBlock(qureg.chunkId, qureg.numAmpsPerChunk, biggerQubit,
                  qureg.numQubitsRepresented);
          pairRank = getChunkOuterBlockPairId(rankIsUpperBiggerQubit, qureg.chunkId, qureg.numAmpsPerChunk,
                  biggerQubit, qureg.numQubitsRepresented);

          exchangePairStateVectorHalves(qureg, pairRank);
          densmatr_mixTwoQubitDepolarisingDistributed(qureg, smallerQubit, biggerQubit, delta, GAMMA_PARTS_1_OR_2);

          // part 3
          compressPairVectorForTwoQubitDepolarise(qureg, smallerQubit, biggerQubit);
          pairRank = getChunkOuterBlockPairIdForPart3(rankIsUpperSmallerQubit, rankIsUpperBiggerQubit,
                  qureg.chunkId, qureg.numAmpsPerChunk, smallerQubit, biggerQubit, qureg.numQubitsRepresented);
          exchangePairStateVectorHalves(qureg, pairRank);
          densmatr_mixTwoQubitDepolarisingDistributed(qureg, smallerQubit, biggerQubit, delta, gamma);

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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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
  const long long int chunkId=qureg.chunkId;

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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
  statevec_controlledCompactUnitaryDistributedKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg.numAmpsPerChunk, 
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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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
  }
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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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
        statevec_pauliYLocal(qureg, targetQubit, conjFac);
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
    statevec_pauliYLocal(qureg, targetQubit, conjFac);
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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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

  qreal recRoot2 = 1.0/sqrt(2);

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
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
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
  //!!need to compare to gpu_local & cpu_local

  qreal stateProb=0, totalStateProb=0;
  int skipValuesWithinRank = halfMatrixBlockFitsInChunk(qureg.numAmpsPerChunk, measureQubit);
  if (skipValuesWithinRank) {
    stateProb = statevec_findProbabilityOfZeroLocal(qureg, measureQubit);
  } else {
    if (!isChunkToSkipInFindPZero(qureg.chunkId, qureg.numAmpsPerChunk, measureQubit)){
      stateProb = statevec_findProbabilityOfZeroDistributed(qureg);
    } else stateProb = 0;
  }
  MPI_Allreduce(&stateProb, &totalStateProb, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (outcome==1) totalStateProb = 1.0 - totalStateProb;
  return totalStateProb;
}

qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome) {

  //!!need to compare to gpu_local & cpu_local

	qreal zeroProb = densmatr_findProbabilityOfZeroLocal(qureg, measureQubit);

	qreal outcomeProb;
	MPI_Allreduce(&zeroProb, &outcomeProb, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
	if (outcome == 1)
		outcomeProb = 1.0 - outcomeProb;

	return outcomeProb;
}

qreal densmatr_calcPurity(Qureg qureg) {

  //!!simple return in cpu_local

  qreal localPurity = densmatr_calcPurityLocal(qureg);

  qreal globalPurity;
  MPI_Allreduce(&localPurity, &globalPurity, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);

  return globalPurity;
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

  //!!cpu_local is similar to gpu_local

  // init MT random number generator with three keys -- time and pid
  // for the MPI version, it is ok that all procs will get the same seed as random numbers will only be
  // used by the master process

  unsigned long int key[2];
  getQuESTDefaultSeedKey(key);
  // this seed will be used to generate the same random number on all procs,
  // therefore we want to make sure all procs receive the same key
  // using cuMPI_UNSIGNED_LONG
  cuMPI_Bcast(key, 2, cuMPI_UINT32_T, 0, cuMPI_COMM_WORLD);
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
          statevec_swapQubitAmps(qureg, targs[t], swapTargs[t]);

  // all target qubits have now been swapped into local memory
  statevec_multiControlledMultiQubitUnitaryLocal(qureg, ctrlMask, swapTargs, numTargs, u);

  // undo swaps
  for (int t=0; t<numTargs; t++)
      if (swapTargs[t] != targs[t])
          statevec_swapQubitAmps(qureg, targs[t], swapTargs[t]);
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

    cudaMalloc(&(qureg->stateVec.real), numAmpsPerRank * sizeof(*(qureg->stateVec.real));
    cudaMalloc(&(qureg->stateVec.imag), numAmpsPerRank * sizeof(*(qureg->stateVec.imag));
    if (env.numRanks > 1) {
      cudaMalloc(&(qureg->stateVec.real), numAmpsPerRank * sizeof(*(qureg->pairStateVec.real)));
      cudaMalloc(&(qureg->stateVec.imag), numAmpsPerRank * sizeof(*(qureg->pairStateVec.imag));
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
  long long int index;
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


//densmatr_mixDephasing(qureg, targetQubit, depolLevel);
//densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephase);
//densmatr_mixTwoQubitDephasing(qureg, qubit1, qubit2, depolLevel);


