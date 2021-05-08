// Some Functions for densmatr

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

// after statevec_findProbabilityOfZeroKernel()
int getNumReductionLevels(long long int numValuesToReduce, int numReducedPerLevel){
  int levels=0;
  while (numValuesToReduce){
      numValuesToReduce = numValuesToReduce/numReducedPerLevel;
      levels++;
  }
  return levels;
}





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



// from double GPU, after statevec_calcInnerProduct()

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


// from double GPU, after copyVecIntoMatrixPairState()
/********************************************/

// !!!!!!!! already have this function before in this file
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

// !!!!!!!! already have this function before in this file
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








// from double GPU, after compressPairVectorForTwoQubitDepolarise()


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




// !!!!!!!! already have this function before in this file
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

// !!!!!!!! already have this function before in this file
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

// !!!!!!!! already have this function before in this file
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



// From double GPU, after statevec_calcProbOfOutcome()

// !!!!!!!! already have this function before in this file
qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome) {

  //!!need to compare to gpu_local & cpu_local

	qreal zeroProb = densmatr_findProbabilityOfZeroLocal(qureg, measureQubit);

	qreal outcomeProb;
	MPI_Allreduce(&zeroProb, &outcomeProb, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);
	if (outcome == 1)
		outcomeProb = 1.0 - outcomeProb;

	return outcomeProb;
}

// !!!!!!!! already have this function before in this file
qreal densmatr_calcPurity(Qureg qureg) {

  //!!simple return in cpu_local

  qreal localPurity = densmatr_calcPurityLocal(qureg);

  qreal globalPurity;
  MPI_Allreduce(&localPurity, &globalPurity, 1, MPI_QuEST_REAL, MPI_SUM, MPI_COMM_WORLD);

  return globalPurity;
}




//densmatr_mixDephasing(qureg, targetQubit, depolLevel);
//densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephase);
//densmatr_mixTwoQubitDephasing(qureg, qubit1, qubit2, depolLevel);




// qreal densmatr_calcTotalProbLocal(Qureg qureg) {}
// qreal densmatr_calcTotalProb(Qureg qureg) {}

// qreal densmatr_calcInnerProduct(Qureg a, Qureg b) {}
// qreal densmatr_calcInnerProductLocal(Qureg a, Qureg b) {}


void NOT_USED_AT_ALL densmatr_initPureState(Qureg targetQureg, Qureg copyQureg) {

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