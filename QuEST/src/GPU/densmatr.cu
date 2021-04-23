__global__ void densmatr_initPureStateKernel(
  long long int numPureAmps,
  qreal *targetVecReal, qreal *targetVecImag, 
  qreal *copyVecReal, qreal *copyVecImag) 
{
  // this is a particular index of the pure copyQureg
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=numPureAmps) return;
  
  qreal realRow = copyVecReal[index];
  qreal imagRow = copyVecImag[index];
  for (long long int col=0; col < numPureAmps; col++) {
      qreal realCol =   copyVecReal[col];
      qreal imagCol = - copyVecImag[col]; // minus for conjugation
      targetVecReal[col*numPureAmps + index] = realRow*realCol - imagRow*imagCol;
      targetVecImag[col*numPureAmps + index] = realRow*imagCol + imagRow*realCol;
  }
}

void densmatr_initPureStateLocal(Qureg targetQureg, Qureg copyQureg)
{
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(copyQureg.numAmpsPerChunk)/threadsPerCUDABlock);
  densmatr_initPureStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      copyQureg.numAmpsPerChunk,
      targetQureg.deviceStateVec.real, targetQureg.deviceStateVec.imag,
      copyQureg.deviceStateVec.real,   copyQureg.deviceStateVec.imag);
}

__global__ void densmatr_initPlusStateKernel(long long int stateVecSize, qreal probFactor, qreal *stateVecReal, qreal *stateVecImag){
  long long int index;

  index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index>=stateVecSize) return;

  stateVecReal[index] = probFactor;
  stateVecImag[index] = 0.0;
}

// @@TODO only in QuEST_cpu.c
void densmatr_initPlusState(Qureg qureg)
{
  qreal probFactor = 1.0/((qreal) (1LL << qureg.numQubitsRepresented));
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  densmatr_initPlusStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, 
      probFactor,
      qureg.deviceStateVec.real, 
      qureg.deviceStateVec.imag);
}

__global__ void densmatr_initClassicalStateKernel(
  long long int densityNumElems, 
  qreal *densityReal, qreal *densityImag, 
  long long int densityInd)
{
  // initialise the state to all zeros
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= densityNumElems) return;
  
  densityReal[index] = 0.0;
  densityImag[index] = 0.0;
  
  if (index==densityInd){
      // classical state has probability 1
      densityReal[densityInd] = 1.0;
      densityImag[densityInd] = 0.0;
  }
}

// @@TODO only in QuEST_cpu.c
void densmatr_initClassicalState(Qureg qureg, long long int stateInd)
{
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
  
  // index of the desired state in the flat density matrix
  long long int densityDim = 1LL << qureg.numQubitsRepresented;
  long long int densityInd = (densityDim + 1)*stateInd;
  
  // identical to pure version
  densmatr_initClassicalStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      qureg.numAmpsPerChunk, 
      qureg.deviceStateVec.real, 
      qureg.deviceStateVec.imag, densityInd);
}





// after statevec_collapseToKnownProbOutcomeLocal() 2021.04.22





/** Maps thread ID to a |..0..><..0..| state and then locates |0><1|, |1><0| and |1><1| */
__global__ void densmatr_collapseToKnownProbOutcomeKernel(
  qreal outcomeProb, qreal* vecReal, qreal *vecImag, long long int numBasesToVisit,
  long long int part1, long long int part2, long long int part3, 
  long long int rowBit, long long int colBit, long long int desired, long long int undesired) 
{
  long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (scanInd >= numBasesToVisit) return;
  
  long long int base = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
  
  // renormalise desired outcome
  vecReal[base + desired] /= outcomeProb;
  vecImag[base + desired] /= outcomeProb;
  
  // kill undesired outcome
  vecReal[base + undesired] = 0;
  vecImag[base + undesired] = 0;
  
  // kill |..0..><..1..| states
  vecReal[base + colBit] = 0;
  vecImag[base + colBit] = 0;
  vecReal[base + rowBit] = 0;
  vecImag[base + rowBit] = 0;
}

/** This involves finding |...i...><...j...| states and killing those where i!=j */
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb) {
  
int rowQubit = measureQubit + qureg.numQubitsRepresented;
  
  int colBit = 1LL << measureQubit;
  int rowBit = 1LL << rowQubit;

  long long int numBasesToVisit = qureg.numAmpsPerChunk/4;
long long int part1 = colBit -1;	
long long int part2 = (rowBit >> 1) - colBit;
long long int part3 = numBasesToVisit - (rowBit >> 1);
  
  long long int desired, undesired;
  if (outcome == 0) {
      desired = 0;
      undesired = colBit | rowBit;
  } else {
      desired = colBit | rowBit;
      undesired = 0;
  }
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numBasesToVisit / (qreal) threadsPerCUDABlock);
  densmatr_collapseToKnownProbOutcomeKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      outcomeProb, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numBasesToVisit,
      part1, part2, part3, rowBit, colBit, desired, undesired);
}

__global__ void densmatr_mixDensityMatrixKernel(Qureg combineQureg, qreal otherProb, Qureg otherQureg, long long int numAmpsToVisit) {
  
  long long int ampInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (ampInd >= numAmpsToVisit) return;
  
  combineQureg.deviceStateVec.real[ampInd] *= 1-otherProb;
  combineQureg.deviceStateVec.imag[ampInd] *= 1-otherProb;

  combineQureg.deviceStateVec.real[ampInd] += otherProb*otherQureg.deviceStateVec.real[ampInd];
  combineQureg.deviceStateVec.imag[ampInd] += otherProb*otherQureg.deviceStateVec.imag[ampInd];
}

void densmatr_mixDensityMatrix(Qureg combineQureg, qreal otherProb, Qureg otherQureg) {
  
  long long int numAmpsToVisit = combineQureg.numAmpsPerChunk;
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixDensityMatrixKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      combineQureg, otherProb, otherQureg, numAmpsToVisit
  );
}

/** Called once for every 4 amplitudes in density matrix 
* Works by establishing the |..0..><..0..| state (for its given index) then 
* visiting |..1..><..0..| and |..0..><..1..|. Labels |part1 X pa><rt2 NOT(X) part3|
* From the brain of Simon Benjamin
*/
__global__ void densmatr_mixDephasingKernel(
  qreal fac, qreal* vecReal, qreal *vecImag, long long int numAmpsToVisit,
  long long int part1, long long int part2, long long int part3, 
  long long int colBit, long long int rowBit)
{
  long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (scanInd >= numAmpsToVisit) return;
  
  long long int ampInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
  vecReal[ampInd + colBit] *= fac;
  vecImag[ampInd + colBit] *= fac;
  vecReal[ampInd + rowBit] *= fac;
  vecImag[ampInd + rowBit] *= fac;
}


void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit, qreal dephFac) {
  
  long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;
  
  int rowQubit = targetQubit + qureg.numQubitsRepresented;
  long long int colBit = 1LL << targetQubit;
  long long int rowBit = 1LL << rowQubit;
  
  long long int part1 = colBit - 1;
  long long int part2 = (rowBit >> 1) - colBit;
  long long int part3 = numAmpsToVisit - (rowBit >> 1);
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixDephasingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      dephFac, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numAmpsToVisit,
      part1, part2, part3, colBit, rowBit);
}

void densmatr_mixDephasing(Qureg qureg, const int targetQubit, qreal dephase) {
  
  if (dephase == 0)
      return;
  
  qreal dephFac = 1 - dephase;
  densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephFac);
}

/** Called 12 times for every 16 amplitudes in density matrix 
* Each sums from the |..0..0..><..0..0..| index to visit either
* |..0..0..><..0..1..|,  |..0..0..><..1..0..|,  |..0..0..><..1..1..|,  |..0..1..><..0..0..|
* etc and so on to |..1..1..><..1..0|. Labels |part1 0 part2 0 par><t3 0 part4 0 part5|.
* From the brain of Simon Benjamin
*/
__global__ void densmatr_mixTwoQubitDephasingKernel(
  qreal fac, qreal* vecReal, qreal *vecImag, long long int numBackgroundStates, long long int numAmpsToVisit,
  long long int part1, long long int part2, long long int part3, long long int part4, long long int part5,
  long long int colBit1, long long int rowBit1, long long int colBit2, long long int rowBit2) 
{
  long long int outerInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (outerInd >= numAmpsToVisit) return;
  
  // sets meta in 1...14 excluding 5, 10, creating bit string DCBA for |..D..C..><..B..A|
  int meta = 1 + (outerInd/numBackgroundStates);
  if (meta > 4) meta++;
  if (meta > 9) meta++;
  
  long long int shift = rowBit2*((meta>>3)%2) + rowBit1*((meta>>2)%2) + colBit2*((meta>>1)%2) + colBit1*(meta%2);
  long long int scanInd = outerInd % numBackgroundStates;
  long long int stateInd = (
      shift + 
      (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2) + ((scanInd&part4)<<3) + ((scanInd&part5)<<4));
  
  vecReal[stateInd] *= fac;
  vecImag[stateInd] *= fac;
}

// @TODO is separating these 12 amplitudes really faster than letting every 16th base modify 12 elems?
void densmatr_mixTwoQubitDephasing(Qureg qureg, int qubit1, int qubit2, qreal dephase) {
  
  if (dephase == 0)
      return;
  
  // assumes qubit2 > qubit1
  
  int rowQubit1 = qubit1 + qureg.numQubitsRepresented;
  int rowQubit2 = qubit2 + qureg.numQubitsRepresented;
  
  long long int colBit1 = 1LL << qubit1;
  long long int rowBit1 = 1LL << rowQubit1;
  long long int colBit2 = 1LL << qubit2;
  long long int rowBit2 = 1LL << rowQubit2;
  
  long long int part1 = colBit1 - 1;
  long long int part2 = (colBit2 >> 1) - colBit1;
  long long int part3 = (rowBit1 >> 2) - (colBit2 >> 1);
  long long int part4 = (rowBit2 >> 3) - (rowBit1 >> 2);
  long long int part5 = (qureg.numAmpsPerChunk/16) - (rowBit2 >> 3);
  qreal dephFac = 1 - dephase;
  
  // refers to states |a 0 b 0 c><d 0 e 0 f| (target qubits are fixed)
  long long int numBackgroundStates = qureg.numAmpsPerChunk/16;
  
  // 12 of these states experience dephasing
  long long int numAmpsToVisit = 12 * numBackgroundStates;
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixTwoQubitDephasingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      dephFac, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numBackgroundStates, numAmpsToVisit,
      part1, part2, part3, part4, part5, colBit1, rowBit1, colBit2, rowBit2);
}

/** Works like mixDephasing but modifies every other element, and elements are averaged in pairs */
__global__ void densmatr_mixDepolarisingKernel(
  qreal depolLevel, qreal* vecReal, qreal *vecImag, long long int numAmpsToVisit,
  long long int part1, long long int part2, long long int part3, 
  long long int bothBits)
{
  long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (scanInd >= numAmpsToVisit) return;
  
  long long int baseInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
  long long int targetInd = baseInd + bothBits;
  
  qreal realAvDepol = depolLevel * 0.5 * (vecReal[baseInd] + vecReal[targetInd]);
  qreal imagAvDepol = depolLevel * 0.5 * (vecImag[baseInd] + vecImag[targetInd]);
  
  vecReal[baseInd]   *= 1 - depolLevel;
  vecImag[baseInd]   *= 1 - depolLevel;
  vecReal[targetInd] *= 1 - depolLevel;
  vecImag[targetInd] *= 1 - depolLevel;
  
  vecReal[baseInd]   += realAvDepol;
  vecImag[baseInd]   += imagAvDepol;
  vecReal[targetInd] += realAvDepol;
  vecImag[targetInd] += imagAvDepol;
}

/** Works like mixDephasing but modifies every other element, and elements are averaged in pairs */
__global__ void densmatr_mixDampingKernel(
  qreal damping, qreal* vecReal, qreal *vecImag, long long int numAmpsToVisit,
  long long int part1, long long int part2, long long int part3, 
  long long int bothBits)
{
  long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (scanInd >= numAmpsToVisit) return;
  
  long long int baseInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
  long long int targetInd = baseInd + bothBits;
  
  qreal realAvDepol = damping  * ( vecReal[targetInd]);
  qreal imagAvDepol = damping  * ( vecImag[targetInd]);
  
  vecReal[targetInd] *= 1 - damping;
  vecImag[targetInd] *= 1 - damping;
  
  vecReal[baseInd]   += realAvDepol;
  vecImag[baseInd]   += imagAvDepol;
}

void densmatr_mixDepolarisingLocal(Qureg qureg, const int targetQubit, qreal depolLevel) {
  
  if (depolLevel == 0)
      return;
  
  densmatr_mixDephasing(qureg, targetQubit, depolLevel);
  
  long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;
  int rowQubit = targetQubit + qureg.numQubitsRepresented;
  
  long long int colBit = 1LL << targetQubit;
  long long int rowBit = 1LL << rowQubit;
  long long int bothBits = colBit | rowBit;
  
  long long int part1 = colBit - 1;
  long long int part2 = (rowBit >> 1) - colBit;
  long long int part3 = numAmpsToVisit - (rowBit >> 1);
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixDepolarisingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      depolLevel, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numAmpsToVisit,
      part1, part2, part3, bothBits);
}

void densmatr_mixDampingLocal(Qureg qureg, const int targetQubit, qreal damping) {
  
  if (damping == 0)
      return;
  
  qreal dephase = sqrt(1-damping);
  densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephase);
  
  long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;
  int rowQubit = targetQubit + qureg.numQubitsRepresented;
  
  long long int colBit = 1LL << targetQubit;
  long long int rowBit = 1LL << rowQubit;
  long long int bothBits = colBit | rowBit;
  
  long long int part1 = colBit - 1;
  long long int part2 = (rowBit >> 1) - colBit;
  long long int part3 = numAmpsToVisit - (rowBit >> 1);
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixDampingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      damping, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numAmpsToVisit,
      part1, part2, part3, bothBits);
}

/** Called once for every 16 amplitudes */
__global__ void densmatr_mixTwoQubitDepolarisingKernel(
  qreal depolLevel, qreal* vecReal, qreal *vecImag, long long int numAmpsToVisit,
  long long int part1, long long int part2, long long int part3, 
  long long int part4, long long int part5,
  long long int rowCol1, long long int rowCol2)
{
  long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
  if (scanInd >= numAmpsToVisit) return;
  
  // index of |..0..0..><..0..0|
  long long int ind00 = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2) + ((scanInd&part4)<<3) + ((scanInd&part5)<<4);
  long long int ind01 = ind00 + rowCol1;
  long long int ind10 = ind00 + rowCol2;
  long long int ind11 = ind00 + rowCol1 + rowCol2;
  
  qreal realAvDepol = depolLevel * 0.25 * (
      vecReal[ind00] + vecReal[ind01] + vecReal[ind10] + vecReal[ind11]);
  qreal imagAvDepol = depolLevel * 0.25 * (
      vecImag[ind00] + vecImag[ind01] + vecImag[ind10] + vecImag[ind11]);
  
  qreal retain = 1 - depolLevel;
  vecReal[ind00] *= retain; vecImag[ind00] *= retain;
  vecReal[ind01] *= retain; vecImag[ind01] *= retain;
  vecReal[ind10] *= retain; vecImag[ind10] *= retain;
  vecReal[ind11] *= retain; vecImag[ind11] *= retain;

  vecReal[ind00] += realAvDepol; vecImag[ind00] += imagAvDepol;
  vecReal[ind01] += realAvDepol; vecImag[ind01] += imagAvDepol;
  vecReal[ind10] += realAvDepol; vecImag[ind10] += imagAvDepol;
  vecReal[ind11] += realAvDepol; vecImag[ind11] += imagAvDepol;
}

void densmatr_mixTwoQubitDepolarisingLocal(Qureg qureg, int qubit1, int qubit2, qreal depolLevel) {
  
  if (depolLevel == 0)
      return;
  
  // assumes qubit2 > qubit1
  
  densmatr_mixTwoQubitDephasing(qureg, qubit1, qubit2, depolLevel);
  
  int rowQubit1 = qubit1 + qureg.numQubitsRepresented;
  int rowQubit2 = qubit2 + qureg.numQubitsRepresented;
  
  long long int colBit1 = 1LL << qubit1;
  long long int rowBit1 = 1LL << rowQubit1;
  long long int colBit2 = 1LL << qubit2;
  long long int rowBit2 = 1LL << rowQubit2;
  
  long long int rowCol1 = colBit1 | rowBit1;
  long long int rowCol2 = colBit2 | rowBit2;
  
  long long int numAmpsToVisit = qureg.numAmpsPerChunk/16;
  long long int part1 = colBit1 - 1;
  long long int part2 = (colBit2 >> 1) - colBit1;
  long long int part3 = (rowBit1 >> 2) - (colBit2 >> 1);
  long long int part4 = (rowBit2 >> 3) - (rowBit1 >> 2);
  long long int part5 = numAmpsToVisit - (rowBit2 >> 3);
  
  int threadsPerCUDABlock, CUDABlocks;
  threadsPerCUDABlock = 128;
  CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
  densmatr_mixTwoQubitDepolarisingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
      depolLevel, qureg.deviceStateVec.real, qureg.deviceStateVec.imag, numAmpsToVisit,
      part1, part2, part3, part4, part5, rowCol1, rowCol2);
}



// after statevec_calcInnerProductLocal() 2021.04.22



/** computes one term of (vec^*T) dens * vec */
__global__ void densmatr_calcFidelityKernel(Qureg dens, Qureg vec, long long int dim, qreal* reducedArray) {

  // figure out which density matrix row to consider
  long long int col;
  long long int row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row >= dim) return;
  
  qreal* densReal = dens.deviceStateVec.real;
  qreal* densImag = dens.deviceStateVec.imag;
  qreal* vecReal  = vec.deviceStateVec.real;
  qreal* vecImag  = vec.deviceStateVec.imag;
  
  // compute the row-th element of the product dens*vec
  qreal prodReal = 0;
  qreal prodImag = 0;
  for (col=0LL; col < dim; col++) {
      qreal densElemReal = densReal[dim*col + row];
      qreal densElemImag = densImag[dim*col + row];
      
      prodReal += densElemReal*vecReal[col] - densElemImag*vecImag[col];
      prodImag += densElemReal*vecImag[col] + densElemImag*vecReal[col];
  }
  
  // multiply with row-th elem of (vec^*)
  qreal termReal = prodImag*vecImag[row] + prodReal*vecReal[row];
  
  // imag of every term should be zero, because each is a valid fidelity calc of an eigenstate
  //qreal termImag = prodImag*vecReal[row] - prodReal*vecImag[row];
  
  extern __shared__ qreal tempReductionArray[];
  tempReductionArray[threadIdx.x] = termReal;
  __syncthreads();
  
  // every second thread reduces
  if (threadIdx.x<blockDim.x/2)
      reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

qreal densmatr_calcFidelityLocal(Qureg qureg, Qureg pureState) {
  
  // we're summing the square of every term in the density matrix
  long long int densityDim = 1LL << qureg.numQubitsRepresented;
  long long int numValuesToReduce = densityDim;
  
  int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
  int maxReducedPerLevel = REDUCE_SHARED_SIZE;
  int firstTime = 1;
  
  while (numValuesToReduce > 1) {
      
      // need less than one CUDA-BLOCK to reduce
      if (numValuesToReduce < maxReducedPerLevel) {
          valuesPerCUDABlock = numValuesToReduce;
          numCUDABlocks = 1;
      }
      // otherwise use only full CUDA-BLOCKS
      else {
          valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
          numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
      }
      // dictates size of reduction array
      sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
      
      // spawn threads to sum the probs in each block
      // store the reduction in the pureState array
      if (firstTime) {
           densmatr_calcFidelityKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
               qureg, pureState, densityDim, pureState.firstLevelReduction);
          firstTime = 0;
          
      // sum the block probs
      } else {
          cudaDeviceSynchronize();    
          copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                  pureState.firstLevelReduction, 
                  pureState.secondLevelReduction, valuesPerCUDABlock); 
          cudaDeviceSynchronize();    
          swapDouble(&(pureState.firstLevelReduction), &(pureState.secondLevelReduction));
      }
      
      numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
  }
  
  qreal fidelity;
  cudaMemcpy(&fidelity, pureState.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
  return fidelity;
}

__global__ void densmatr_calcHilbertSchmidtDistanceSquaredKernel(
  qreal* aRe, qreal* aIm, qreal* bRe, qreal* bIm, 
  long long int numAmpsToSum, qreal *reducedArray
) {
  // figure out which density matrix term this thread is assigned
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= numAmpsToSum) return;
  
  // compute this thread's sum term
  qreal difRe = aRe[index] - bRe[index];
  qreal difIm = aIm[index] - bIm[index];
  qreal term = difRe*difRe + difIm*difIm;
  
  // array of each thread's collected term, to be summed
  extern __shared__ qreal tempReductionArray[];
  tempReductionArray[threadIdx.x] = term;
  __syncthreads();
  
  // every second thread reduces
  if (threadIdx.x<blockDim.x/2)
      reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

/* computes sqrt(Tr( (a-b) conjTrans(a-b) ) = sqrt( sum of abs vals of (a-b)) */
qreal densmatr_calcHilbertSchmidtDistanceSquaredLocal(Qureg a, Qureg b) {
  
  // we're summing the square of every term in (a-b)
  long long int numValuesToReduce = a.numAmpsPerChunk;
  
  int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
  int maxReducedPerLevel = REDUCE_SHARED_SIZE;
  int firstTime = 1;
  
  while (numValuesToReduce > 1) {
      
      // need less than one CUDA-BLOCK to reduce
      if (numValuesToReduce < maxReducedPerLevel) {
          valuesPerCUDABlock = numValuesToReduce;
          numCUDABlocks = 1;
      }
      // otherwise use only full CUDA-BLOCKS
      else {
          valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
          numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
      }
      // dictates size of reduction array
      sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
      
      // spawn threads to sum the probs in each block (store reduction temp values in a's reduction array)
      if (firstTime) {
           densmatr_calcHilbertSchmidtDistanceSquaredKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
               a.deviceStateVec.real, a.deviceStateVec.imag, 
               b.deviceStateVec.real, b.deviceStateVec.imag, 
               numValuesToReduce, a.firstLevelReduction);
          firstTime = 0;
          
      // sum the block probs
      } else {
          cudaDeviceSynchronize();    
          copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                  a.firstLevelReduction, 
                  a.secondLevelReduction, valuesPerCUDABlock); 
          cudaDeviceSynchronize();    
          swapDouble(&(a.firstLevelReduction), &(a.secondLevelReduction));
      }
      
      numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
  }
  
  qreal trace;
  cudaMemcpy(&trace, a.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
  
  // modified by yh 2020.03.19
  // qreal sqrtTrace = sqrt(trace);
  // return sqrtTrace;
  return trace;
}

__global__ void densmatr_calcPurityKernel(qreal* vecReal, qreal* vecImag, long long int numAmpsToSum, qreal *reducedArray) {
  
  // figure out which density matrix term this thread is assigned
  long long int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= numAmpsToSum) return;
  
  qreal term = vecReal[index]*vecReal[index] + vecImag[index]*vecImag[index];
  
  // array of each thread's collected probability, to be summed
  extern __shared__ qreal tempReductionArray[];
  tempReductionArray[threadIdx.x] = term;
  __syncthreads();
  
  // every second thread reduces
  if (threadIdx.x<blockDim.x/2)
      reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

/** Computes the trace of the density matrix squared */
qreal densmatr_calcPurityLocal(Qureg qureg) {
  
  // we're summing the square of every term in the density matrix
  long long int numValuesToReduce = qureg.numAmpsPerChunk;
  
  int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
  int maxReducedPerLevel = REDUCE_SHARED_SIZE;
  int firstTime = 1;
  
  while (numValuesToReduce > 1) {
      
      // need less than one CUDA-BLOCK to reduce
      if (numValuesToReduce < maxReducedPerLevel) {
          valuesPerCUDABlock = numValuesToReduce;
          numCUDABlocks = 1;
      }
      // otherwise use only full CUDA-BLOCKS
      else {
          valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
          numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
      }
      // dictates size of reduction array
      sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
      
      // spawn threads to sum the probs in each block
      if (firstTime) {
           densmatr_calcPurityKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
               qureg.deviceStateVec.real, qureg.deviceStateVec.imag, 
               numValuesToReduce, qureg.firstLevelReduction);
          firstTime = 0;
          
      // sum the block probs
      } else {
          cudaDeviceSynchronize();    
          copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                  qureg.firstLevelReduction, 
                  qureg.secondLevelReduction, valuesPerCUDABlock); 
          cudaDeviceSynchronize();    
          swapDouble(&(qureg.firstLevelReduction), &(qureg.secondLevelReduction));
      }
      
      numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
  }
  
  qreal traceDensSquared;
  cudaMemcpy(&traceDensSquared, qureg.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
  return traceDensSquared;
}



// after statevec_calcProbOfOutcomeLocal() 2021.04.22



qreal densmatr_calcProbOfOutcomeLocal(Qureg qureg, const int measureQubit, int outcome)
{
    qreal outcomeProb = densmatr_findProbabilityOfZeroLocal(qureg, measureQubit);
    if (outcome==1) 
        outcomeProb = 1.0 - outcomeProb;
    return outcomeProb;
}

/** computes Tr(conjTrans(a) b) = sum of (a_ij^* b_ij), which is a real number */
__global__ void densmatr_calcInnerProductKernel(
    Qureg a, Qureg b, long long int numTermsToSum, qreal* reducedArray
) {    
    long long int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numTermsToSum) return;
    
    // Re{ conj(a) b } = Re{ (aRe - i aIm)(bRe + i bIm) } = aRe bRe + aIm bIm
    qreal prod = (
          a.deviceStateVec.real[index]*b.deviceStateVec.real[index] 
        + a.deviceStateVec.imag[index]*b.deviceStateVec.imag[index]);
    
    // array of each thread's collected sum term, to be summed
    extern __shared__ qreal tempReductionArray[];
    tempReductionArray[threadIdx.x] = prod;
    __syncthreads();
    
    // every second thread reduces
    if (threadIdx.x<blockDim.x/2)
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

qreal densmatr_calcInnerProductLocal(Qureg a, Qureg b) {
    
    // we're summing the square of every term in the density matrix
    long long int numValuesToReduce = a.numAmpsTotal;
    
    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;
    int firstTime = 1;
    
    while (numValuesToReduce > 1) {
        
        // need less than one CUDA-BLOCK to reduce
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        // otherwise use only full CUDA-BLOCKS
        else {
            valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        // dictates size of reduction array
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
        
        // spawn threads to sum the terms in each block
        // arbitrarily store the reduction in the b qureg's array
        if (firstTime) {
             densmatr_calcInnerProductKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                 a, b, a.numAmpsTotal, b.firstLevelReduction);
            firstTime = 0;
        }    
        // sum the block terms
        else {
            cudaDeviceSynchronize();    
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    b.firstLevelReduction, 
                    b.secondLevelReduction, valuesPerCUDABlock); 
            cudaDeviceSynchronize();    
            swapDouble(&(b.firstLevelReduction), &(b.secondLevelReduction));
        }
        
        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }
    
    qreal innerprod;
    cudaMemcpy(&innerprod, b.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return innerprod;
}



// after swapDouble() 2021.04.22



qreal densmatr_findProbabilityOfZero(Qureg qureg, const int measureQubit)
{
    long long int densityDim = 1LL << qureg.numQubitsRepresented;
    long long int numValuesToReduce = densityDim >> 1;  // half of the diagonal has measureQubit=0
    
    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;
    int firstTime = 1;
    
    while (numValuesToReduce > 1) {
        
        // need less than one CUDA-BLOCK to reduce
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        // otherwise use only full CUDA-BLOCKS
        else {
            valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
        
        // spawn threads to sum the probs in each block
        if (firstTime) {
            densmatr_findProbabilityOfZeroKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                qureg, measureQubit, qureg.firstLevelReduction);
            firstTime = 0;
            
        // sum the block probs
        } else {
            cudaDeviceSynchronize();    
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    qureg.firstLevelReduction, 
                    qureg.secondLevelReduction, valuesPerCUDABlock); 
            cudaDeviceSynchronize();    
            swapDouble(&(qureg.firstLevelReduction), &(qureg.secondLevelReduction));
        }
        
        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }
    
    qreal zeroProb;
    cudaMemcpy(&zeroProb, qureg.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return zeroProb;
}



// after copySharedReduceBlock() 2021.04.22



__global__ void densmatr_findProbabilityOfZeroKernel(
  Qureg qureg, const int measureQubit, qreal *reducedArray
) {
  // run by each thread
  // use of block here refers to contiguous amplitudes where measureQubit = 0, 
  // (then =1) and NOT the CUDA block, which is the partitioning of CUDA threads
  
  long long int densityDim    = 1LL << qureg.numQubitsRepresented;
  long long int numTasks      = densityDim >> 1;
  long long int sizeHalfBlock = 1LL << (measureQubit);
  long long int sizeBlock     = 2LL * sizeHalfBlock;
  
  long long int thisBlock;    // which block this thread is processing
  long long int thisTask;     // which part of the block this thread is processing
  long long int basisIndex;   // index of this thread's computational basis state
  long long int densityIndex; // " " index of |basis><basis| in the flat density matrix
  
  // array of each thread's collected probability, to be summed
  extern __shared__ qreal tempReductionArray[];
  
  // figure out which density matrix prob that this thread is assigned
  thisTask = blockIdx.x*blockDim.x + threadIdx.x;
  if (thisTask>=numTasks) return;
  thisBlock = thisTask / sizeHalfBlock;
  basisIndex = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
  densityIndex = (densityDim + 1) * basisIndex;
  
  // record the probability in the CUDA-BLOCK-wide array
  qreal prob = qureg.deviceStateVec.real[densityIndex];   // im[densityIndex] assumed ~ 0
  tempReductionArray[threadIdx.x] = prob;
  
  // sum the probs collected by this CUDA-BLOCK's threads into a per-CUDA-BLOCK array
  __syncthreads();
  if (threadIdx.x<blockDim.x/2){
      reduceBlock(tempReductionArray, reducedArray, blockDim.x);
  }
}



// after statevec_multiRotateZ() 2021.04.22



qreal densmatr_calcTotalProbLocal(Qureg qureg) {
    
  // computes the trace using Kahan summation
  qreal pTotal=0;
  qreal y, t, c;
  c = 0;
  
  long long int numCols = 1LL << qureg.numQubitsRepresented;
  long long diagIndex;
  
  copyStateFromGPU(qureg);
  
  for (int col=0; col< numCols; col++) {
      diagIndex = col*(numCols + 1);
      y = qureg.stateVec.real[diagIndex] - c;
      t = pTotal + y;
      c = ( t - pTotal ) - y; // brackets are important
      pTotal = t;
  }
  
  return pTotal;
}