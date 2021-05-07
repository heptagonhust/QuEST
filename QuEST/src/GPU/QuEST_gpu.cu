#include "QuEST_gpu_internal.h"

ArgMatrix2 argifyMatrix2(ComplexMatrix2 m) {    
  ArgMatrix2 a;
  a.r0c0.real=m.real[0][0]; a.r0c0.imag=m.imag[0][0];
  a.r0c1.real=m.real[0][1]; a.r0c1.imag=m.imag[0][1];
  a.r1c0.real=m.real[1][0]; a.r1c0.imag=m.imag[1][0];
  a.r1c1.real=m.real[1][1]; a.r1c1.imag=m.imag[1][1];
  return a;
}

ArgMatrix4 argifyMatrix4(ComplexMatrix4 m) {     
  ArgMatrix4 a;
  a.r0c0.real=m.real[0][0]; a.r0c0.imag=m.imag[0][0];
  a.r0c1.real=m.real[0][1]; a.r0c1.imag=m.imag[0][1];
  a.r0c2.real=m.real[0][2]; a.r0c2.imag=m.imag[0][2];
  a.r0c3.real=m.real[0][3]; a.r0c3.imag=m.imag[0][3];
  a.r1c0.real=m.real[1][0]; a.r1c0.imag=m.imag[1][0];
  a.r1c1.real=m.real[1][1]; a.r1c1.imag=m.imag[1][1];
  a.r1c2.real=m.real[1][2]; a.r1c2.imag=m.imag[1][2];
  a.r1c3.real=m.real[1][3]; a.r1c3.imag=m.imag[1][3];
  a.r2c0.real=m.real[2][0]; a.r2c0.imag=m.imag[2][0];
  a.r2c1.real=m.real[2][1]; a.r2c1.imag=m.imag[2][1];
  a.r2c2.real=m.real[2][2]; a.r2c2.imag=m.imag[2][2];
  a.r2c3.real=m.real[2][3]; a.r2c3.imag=m.imag[2][3];
  a.r3c0.real=m.real[3][0]; a.r3c0.imag=m.imag[3][0];
  a.r3c1.real=m.real[3][1]; a.r3c1.imag=m.imag[3][1];
  a.r3c2.real=m.real[3][2]; a.r3c2.imag=m.imag[3][2];
  a.r3c3.real=m.real[3][3]; a.r3c3.imag=m.imag[3][3];
  return a;
}

void swapDouble(qreal **a, qreal **b){
  qreal *temp;
  temp = *a;
  *a = *b;
  *b = temp;
}

#ifdef __cplusplus
extern "C" {
#endif

//
// Derived from old single GPU version
// i.e. compared with CPU version, these functions only exist
// in QuEST_cpu.c, not in QuEST_cpu_local.c, and there are no
// related `*Local`-suffix functions for them.
//

__global__ void statevec_initDebugStateKernel(
  long long int chunkSize, 
  long long int chunkId, 
  qreal *stateVecReal, 
  qreal *stateVecImag)
{

  long long int index;
  long long int indexOffset = chunkSize * chunkId;

  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=chunkSize) return;

  stateVecReal[index] = ((indexOffset + index)*2.0)/10.0;
  stateVecImag[index] = ((indexOffset + index)*2.0+1.0)/10.0;
}

void statevec_initDebugState(Qureg qureg)
{
  // stage 1 done!

  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_initDebugStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk,
      qureg.chunkId,
      qureg.stateVec.real, 
      qureg.stateVec.imag);
}


__global__ void statevec_initStateOfSingleQubitKernel(
  long long int chunkSize,
  long long int numChunks,
  long long int chunkId,
  qreal *stateVecReal, 
  qreal *stateVecImag, 
  int qubitId, 
  int outcome)
{

  long long int index;
  long long int stateVecSize = chunkSize*numChunks;

  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=chunkSize) return;

  qreal normFactor = 1.0/sqrt((qreal)stateVecSize/2);

  int bit = extractBit(qubitId, index+chunkId*chunkSize);
  if (bit==outcome) {
      stateVecReal[index] = normFactor;
      stateVecImag[index] = 0.0;
  } else {
      stateVecReal[index] = 0.0;
      stateVecImag[index] = 0.0;
  }
}

void statevec_initStateOfSingleQubit(Qureg *qureg, int qubitId, int outcome)
{
  // stage 1 done!

  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(qureg->numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_initStateOfSingleQubitKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
    qureg->numAmpsPerChunk, 
    qureg->numChunks,
    qureg->chunkId,
    qureg->stateVec.real, 
    qureg->stateVec.imag, 
    qubitId, outcome);
}

__global__ void statevec_compareStatesKernel(
  long long int chunkSize,
  qreal& mq1Real, 
  qreal& mq2Real,
  qreal& mq1Imag,
  qreal& mq2Imag,
  qreal precision,
  int *flag)
{
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= chunkSize) return ;
  if (absReal(mq1Real - mq2Real) > precision ||
      absReal(mq1Imag - mq2Imag) > precision)
    {
      *flag = 1;
      return ;
    }
}

int statevec_compareStates(Qureg mq1, Qureg mq2, qreal precision)
{
  // stage 1 done!

  long long int chunkSize = mq1.numAmpsPerChunk;

  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(chunkSize)/threadsPerCUDABlock);
  
  int *d_flag_ptr = (int*)mallocZeroVarInDevice(1);
  for (long long int i=0; i<chunkSize; i++){
    statevec_compareStatesKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      chunkSize,
      mq1.stateVec.real[i],
      mq2.stateVec.real[i],
      mq1.stateVec.imag[i],
      mq2.stateVec.imag[i],
      precision,
      d_flag_ptr);
  }
  int h_flag;
  cudaMemcpy(&h_flag, d_flag_ptr, 1, cudaMemcpyDeviceToHost);
  return 1 - h_flag;
}


__global__ void statevec_phaseShiftByTermKernel(Qureg qureg, const int targetQubit, qreal cosAngle, qreal sinAngle) {
  // stage 1 done!

  // !only for single gpu
  // long long int sizeBlock, sizeHalfBlock, thisBlock;
  // long long int indexUp, indexLo;

  qreal stateRealLo, stateImagLo;
  long long int thisTask, exactTask; // exactTask is global rank for distributed gpu.
  // const long long int numTasks = qureg.numAmpsPerChunk >> 1; // !only for single gpu
  const long long int numTasks = qureg.numAmpsPerChunk;

  // distributed gpu
  const long long int sizeChunk = qureg.numAmpsPerChunk;
  const long long int chunkId = qureg.chunkId;

  /* yh comment */
  /* sizeHalfBlock & sizeBlock using binary count trick
      e.g. qubit num = 3, target qubit = 1 (id begin with 0)
      000, 001 the center bit (qubit1) is 0 occur continuously for 2 times
      010, 011 when the center bit is 1, the next related bit(#) will add sizeBlock(=4) to index
      100, 101
      #110, 111
  */
  // !only for single gpu
  // sizeHalfBlock = 1LL << targetQubit;
  // sizeBlock     = 2LL * sizeHalfBlock;

  qreal *stateVecReal = qureg.stateVec.real;
  qreal *stateVecImag = qureg.stateVec.imag;

  // thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  // if (thisTask>=numTasks) return;
  // thisBlock   = thisTask / sizeHalfBlock;
  // indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
  // indexLo     = indexUp + sizeHalfBlock;
  
  thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  if (thisTask>=numTasks) return;
  exactTask = thisTask + chunkId*sizeChunk;

  if ( extractBit(targetQubit, exactTask) ) {
      
      stateRealLo = stateVecReal[thisTask];
      stateImagLo = stateVecImag[thisTask];

      stateVecReal[thisTask] = cosAngle*stateRealLo - sinAngle*stateImagLo;
      stateVecImag[thisTask] = sinAngle*stateRealLo + cosAngle*stateImagLo;
  }
}

void statevec_phaseShiftByTerm(Qureg qureg, const int targetQubit, Complex term)
{   
  // stage 1 done!
  // chunkId done!
  
  qreal cosAngle = term.real;
  qreal sinAngle = term.imag;
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  
  // CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_phaseShiftByTermKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, cosAngle, sinAngle);
}


__global__ void statevec_controlledPhaseShiftKernel(Qureg qureg, const int idQubit1, const int idQubit2, qreal cosAngle, qreal sinAngle)
{
  long long int index;
  long long int stateVecSize;
  int bit1, bit2;

  const long long int chunkSize = qureg.numAmpsPerChunk;
  const long long int chunkId=qureg.chunkId;
  
  qreal stateRealLo, stateImagLo;

  stateVecSize = qureg.numAmpsPerChunk;
  qreal *stateVecReal = qureg.stateVec.real;
  qreal *stateVecImag = qureg.stateVec.imag;

  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;

  bit1 = extractBit (idQubit1, index+chunkId*chunkSize);
  bit2 = extractBit (idQubit2, index+chunkId*chunkSize);
  if (bit1 && bit2) {
      stateRealLo = stateVecReal[index];
      stateImagLo = stateVecImag[index];
      
      stateVecReal[index] = cosAngle*stateRealLo - sinAngle*stateImagLo;
      stateVecImag[index] = sinAngle*stateRealLo + cosAngle*stateImagLo;
  }
}

void statevec_controlledPhaseShift(Qureg qureg, const int idQubit1, const int idQubit2, qreal angle)
{
  // stage 1 done!

  qreal cosAngle = cos(angle);
  qreal sinAngle = sin(angle);
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_controlledPhaseShiftKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, idQubit1, idQubit2, cosAngle, sinAngle);
}


__global__ void statevec_multiControlledPhaseShiftKernel(Qureg qureg, long long int mask, qreal cosAngle, qreal sinAngle) {
  qreal stateRealLo, stateImagLo;
  long long int index;
  long long int stateVecSize;

  const long long int chunkSize=qureg.numAmpsPerChunk;
  const long long int chunkId=qureg.chunkId;
  
  stateVecSize = qureg.numAmpsPerChunk;
  qreal *stateVecReal = qureg.stateVec.real;
  qreal *stateVecImag = qureg.stateVec.imag;
  
  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;

  if (mask == (mask & (index+chunkId*chunkSize)) ){
      stateRealLo = stateVecReal[index];
      stateImagLo = stateVecImag[index];
      stateVecReal[index] = cosAngle*stateRealLo - sinAngle*stateImagLo;
      stateVecImag[index] = sinAngle*stateRealLo + cosAngle*stateImagLo;
  }
}

void statevec_multiControlledPhaseShift(Qureg qureg, int *controlQubits, int numControlQubits, qreal angle)
{
  // stage 1 done!

  qreal cosAngle = cos(angle);
  qreal sinAngle = sin(angle);

  long long int mask = getQubitBitMask(controlQubits, numControlQubits);
      
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_multiControlledPhaseShiftKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask, cosAngle, sinAngle);
}


__global__ void statevec_multiRotateZKernel(Qureg qureg, long long int mask, qreal cosAngle, qreal sinAngle) {
  
  long long int stateVecSize = qureg.numAmpsPerChunk;
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;
  
  const long long int chunkSize=qureg.numAmpsPerChunk;
  const long long int chunkId=qureg.chunkId;
  
  qreal *stateVecReal = qureg.stateVec.real;
  qreal *stateVecImag = qureg.stateVec.imag;
  
  // odd-parity target qubits get fac_j = -1
  int fac = getBitMaskParity(mask & (index+chunkId*chunkSize))? -1 : 1;
  qreal stateReal = stateVecReal[index];
  qreal stateImag = stateVecImag[index];
  
  stateVecReal[index] = cosAngle*stateReal + fac * sinAngle*stateImag;
  stateVecImag[index] = - fac * sinAngle*stateReal + cosAngle*stateImag;  
}

void statevec_multiRotateZ(Qureg qureg, long long int mask, qreal angle)
{   
  // stage 1 done!

  qreal cosAngle = cos(angle/2.0);
  qreal sinAngle = sin(angle/2.0);
      
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  statevec_multiRotateZKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask, cosAngle, sinAngle);
}


__global__ void statevec_controlledPhaseFlipKernel(Qureg qureg, const int idQubit1, const int idQubit2)
{
    long long int index;
    long long int stateVecSize;
    int bit1, bit2;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    stateVecSize = qureg.numAmpsPerChunk;
    qreal *stateVecReal = qureg.stateVec.real;
    qreal *stateVecImag = qureg.stateVec.imag;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    bit1 = extractBit (idQubit1, index+chunkId*chunkSize);
    bit2 = extractBit (idQubit2, index+chunkId*chunkSize);
    if (bit1 && bit2) {
        stateVecReal [index] = - stateVecReal [index];
        stateVecImag [index] = - stateVecImag [index];
    }
}

void statevec_controlledPhaseFlip(Qureg qureg, const int idQubit1, const int idQubit2)
{
    // stage 1 done!
    
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_controlledPhaseFlipKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, idQubit1, idQubit2);
}


__global__ void statevec_multiControlledPhaseFlipKernel(Qureg qureg, long long int mask)
{
    long long int index;
    long long int stateVecSize;

    stateVecSize = qureg.numAmpsPerChunk;
    qreal *stateVecReal = qureg.stateVec.real;
    qreal *stateVecImag = qureg.stateVec.imag;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    if (mask == (mask & index) ){
        stateVecReal [index] = - stateVecReal [index];
        stateVecImag [index] = - stateVecImag [index];
    }
}

void statevec_multiControlledPhaseFlip(Qureg qureg, int *controlQubits, int numControlQubits)
{
    int threadsPerCUDABlock, CUDABlocks;
    long long int mask = getQubitBitMask(controlQubits, numControlQubits);
    threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_multiControlledPhaseFlipKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask);
}


__global__ void statevec_setWeightedQuregKernel(Complex fac1, Qureg qureg1, Complex fac2, Qureg qureg2, Complex facOut, Qureg out) {

  long long int ampInd = blockIdx.x*blockDim.x + threadIdx.x;
  long long int numAmpsToVisit = qureg1.numAmpsPerChunk;
  if (ampInd >= numAmpsToVisit) return;

  qreal *vecRe1 = qureg1.stateVec.real;
  qreal *vecIm1 = qureg1.stateVec.imag;
  qreal *vecRe2 = qureg2.stateVec.real;
  qreal *vecIm2 = qureg2.stateVec.imag;
  qreal *vecReOut = out.stateVec.real;
  qreal *vecImOut = out.stateVec.imag;

  qreal facRe1 = fac1.real; 
  qreal facIm1 = fac1.imag;
  qreal facRe2 = fac2.real;
  qreal facIm2 = fac2.imag;
  qreal facReOut = facOut.real;
  qreal facImOut = facOut.imag;

  qreal re1,im1, re2,im2, reOut,imOut;
  long long int index = ampInd;

  re1 = vecRe1[index]; im1 = vecIm1[index];
  re2 = vecRe2[index]; im2 = vecIm2[index];
  reOut = vecReOut[index];
  imOut = vecImOut[index];

  vecReOut[index] = (facReOut*reOut - facImOut*imOut) + (facRe1*re1 - facIm1*im1) + (facRe2*re2 - facIm2*im2);
  vecImOut[index] = (facReOut*imOut + facImOut*reOut) + (facRe1*im1 + facIm1*re1) + (facRe2*im2 + facIm2*re2);
}

void statevec_setWeightedQureg(Complex fac1, Qureg qureg1, Complex fac2, Qureg qureg2, Complex facOut, Qureg out) {

  long long int numAmpsToVisit = qureg1.numAmpsPerChunk;

  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = DEFAULT_THREADS_PER_BLOCK;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  statevec_setWeightedQuregKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      fac1, qureg1, fac2, qureg2, facOut, out
  );
}



// densmatr
void densmatr_initPureState(Qureg targetQureg, Qureg copyQureg){}
void densmatr_initPlusState(Qureg qureg){}
void densmatr_initClassicalState(Qureg qureg, long long int stateInd){}
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb){}
void densmatr_mixDensityMatrix(Qureg combineQureg, qreal otherProb, Qureg otherQureg){}
void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit, qreal dephFac){}
void densmatr_mixDephasing(Qureg qureg, const int targetQubit, qreal dephase){}
void densmatr_mixTwoQubitDephasing(Qureg qureg, int qubit1, int qubit2, qreal dephase){}
void densmatr_mixDepolarising(Qureg qureg, const int targetQubit, qreal depolLevel){}
void densmatr_mixDamping(Qureg qureg, const int targetQubit, qreal damping){}
void densmatr_mixTwoQubitDepolarising(Qureg qureg, int qubit1, int qubit2, qreal depolLevel){}
qreal densmatr_calcFidelity(Qureg qureg, Qureg pureState){return (qreal)0;}
qreal densmatr_calcHilbertSchmidtDistanceSquared(Qureg a, Qureg b){return (qreal)0;}
qreal densmatr_calcPurity(Qureg qureg){return (qreal)0;}
qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome){return (qreal)0;}
qreal densmatr_findProbabilityOfZero(Qureg qureg, const int measureQubit){return (qreal)0;}
qreal densmatr_calcTotalProb(Qureg qureg){return (qreal)0;}
qreal densmatr_calcHilbertSchmidtDistance(Qureg a, Qureg b){return (qreal)0;}
qreal densmatr_calcInnerProduct(Qureg a, Qureg b){return (qreal)0;}


#ifdef __cplusplus
}
#endif
