#ifndef _QUEST_GPU_H_
#define _QUEST_GPU_H_

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include <assert.h>

// Distributed under MIT licence. See https://github.com/QuEST-Kit/QuEST/blob/master/LICENCE.txt for details

/** @file
 * An implementation of the backend in ../QuEST_internal.h for a GPU environment.
 *
 * @author Ania Brown 
 * @author Tyson Jones
 */

# include "QuEST.h"
# include "QuEST_precision.h"
# include "QuEST_internal.h"    // purely to resolve getQuESTDefaultSeedKey
# include "mt19937ar.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

# define REDUCE_SHARED_SIZE 512
# define DEBUG 0


/*
 * struct types for concisely passing unitaries to kernels
 */
 
 // hide these from doxygen
 /// \cond HIDDEN_SYMBOLS  
 
 typedef struct ArgMatrix2 {
     Complex r0c0, r0c1;
     Complex r1c0, r1c1;
 } ArgMatrix2;
 
 typedef struct ArgMatrix4
 {
     Complex r0c0, r0c1, r0c2, r0c3;
     Complex r1c0, r1c1, r1c2, r1c3;
     Complex r2c0, r2c1, r2c2, r2c3;
     Complex r3c0, r3c1, r3c2, r3c3;
 } ArgMatrix4;
 
ArgMatrix2 argifyMatrix2(ComplexMatrix2 m);
ArgMatrix4 argifyMatrix4(ComplexMatrix4 m);
 
 /// \endcond


/*
* Bit twiddling functions are defined seperately here in the CPU backend, 
* since the GPU backend  needs a device-specific redefinition to be callable 
* from GPU kernels. These are called in both QuEST_cpu and QuEST_cpu_distributed 
* and defined in here since public inline methods in C must go in the header
*/

// inline int extractBitOnCPU (const int locationOfBitFromRight, const long long int theEncodedNumber);
// inline long long int flipBitOnCPU(long long int number, int bitInd);
// inline int maskContainsBitOnCPU(long long int mask, int bitInd);

inline int extractBitOnCPU (const int locationOfBitFromRight, const long long int theEncodedNumber) {
  return (theEncodedNumber & ( 1LL << locationOfBitFromRight )) >> locationOfBitFromRight;
}

inline long long int flipBitOnCPU(long long int number, int bitInd) {
  return (number ^ (1LL << bitInd));
}

inline int maskContainsBitOnCPU(long long int mask, int bitInd) {
  return mask & (1LL << bitInd);
}

// inline int isOddParityOnCPU(long long int number, int qb1, int qb2) {
//     return extractBitOnCPU(qb1, number) != extractBitOnCPU(qb2, number);
// }

// inline long long int insertZeroBitOnCPU(long long int number, int index) {
//     long long int left, right;
//     left = (number >> index) << index;
//     right = number - left;
//     return (left << 1) ^ right;
// }

// inline long long int insertTwoZeroBitsOnCPU(long long int number, int bit1, int bit2) {
//     int small = (bit1 < bit2)? bit1 : bit2;
//     int big = (bit1 < bit2)? bit2 : bit1;
//     return insertZeroBitOnCPU(insertZeroBitOnCPU(number, small), big);
// }


/*
 * in-kernel bit twiddling functions
 */

 __forceinline__ __device__ int extractBit (int locationOfBitFromRight, long long int theEncodedNumber) {
  return (theEncodedNumber & ( 1LL << locationOfBitFromRight )) >> locationOfBitFromRight;
}

__forceinline__ __device__ int getBitMaskParity(long long int mask) {
  int parity = 0;
  while (mask) {
      parity = !parity;
      mask = mask & (mask-1);
  }
  return parity;
}

__forceinline__ __device__ long long int flipBit(long long int number, int bitInd) {
  return (number ^ (1LL << bitInd));
}

__forceinline__ __device__ long long int insertZeroBit(long long int number, int index) {
  long long int left, right;
  left = (number >> index) << index;
  right = number - left;
  return (left << 1) ^ right;
}

__forceinline__ __device__ long long int insertTwoZeroBits(long long int number, int bit1, int bit2) {
  int small = (bit1 < bit2)? bit1 : bit2;
  int big = (bit1 < bit2)? bit2 : bit1;
  return insertZeroBit(insertZeroBit(number, small), big);
}

__forceinline__ __device__ long long int insertZeroBits(long long int number, int* inds, int numInds) {
  /* inserted bit inds must strictly increase, so that their final indices are correct.
   * in-lieu of sorting (avoided since no C++ variable-size arrays, and since we're already 
   * memory bottle-necked so overhead eats this slowdown), we find the next-smallest index each 
   * at each insert. recall every element of inds (a positive or zero number) is unique.
   * This function won't appear in the CPU code, which can use C99 variable-size arrays and 
   * ought to make a sorted array before threading
   */
   int curMin = inds[0];
   int prevMin = -1;
   for (int n=0; n < numInds; n++) {
       
       // find next min
       for (int t=0; t < numInds; t++)
          if (inds[t]>prevMin && inds[t]<curMin)
              curMin = inds[t];
      
      number = insertZeroBit(number, curMin);
      
      // set curMin to an arbitrary non-visited elem
      prevMin = curMin;
      for (int t=0; t < numInds; t++)
          if (inds[t] > curMin) {
              curMin = inds[t];
              break;
          }
   }
   return number;
}


/* None side-effects Functions only for numerical calculation */

// copy from distributed CPU, for `statevec_getRealAmp`
// int getChunkIdFromIndex(Qureg qureg, long long int index){
//     return index/qureg.numAmpsPerChunk; // this is numAmpsPerChunk
// }

void swapDouble(qreal **a, qreal **b);

/* Inline tools for GPU version */

inline void setRealInDevice(qreal *d_ptr, qreal *h_ptr) {
  cudaDeviceSynchronize();
  cudaMemcpy(d_ptr, h_ptr, sizeof(qreal), cudaMemcpyHostToDevice);
}

inline qreal getRealInDevice(qreal *d_ptr) {
  cudaDeviceSynchronize();
  qreal ret = 0;
  cudaMemcpy(&ret, d_ptr, sizeof(qreal), cudaMemcpyDeviceToHost);
  return ret;
}

inline qreal* mallocZeroRealInDevice(size_t count) {
  qreal *ret = NULL;
  cudaMalloc(&ret, count);
  cudaMemset(ret, 0, count);
  return ret;
}

inline void freeRealInDevice(qreal *d_ptr) {
  cudaFree(d_ptr);
}

inline void setVarInDevice(void *d_ptr, void *h_ptr, size_t size) {
  cudaDeviceSynchronize();
  cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
}

inline int getIntInDevice(void *d_ptr) {
  cudaDeviceSynchronize();
  int ret = 0;
  cudaMemcpy(&ret, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  return ret;
}

inline void* mallocZeroVarInDevice(size_t count) {
  void *ret = NULL;
  cudaMalloc(&ret, count);
  cudaMemset(ret, 0, count);
  return ret;
}

inline void freeVarInDevice(void *d_ptr) {
  cudaFree(d_ptr);
}

inline int isReadyOnGPU(Qureg qureg) {
  return 1;
}

template <typename T>
inline void displayDeviceVarOnHost(T *ptr_begin, T *ptr_end) {
  thrust::device_ptr<T> dev_ptr_begin = thrust::device_pointer_cast(ptr_begin);
  thrust::device_ptr<T> dev_ptr_end = thrust::device_pointer_cast(ptr_end);
  thrust::device_vector<T> d_x(dev_ptr_begin, dev_ptr_end);
  for(int i = 0; i < d_x.size(); i++)
    std::cout << (T)d_x[i] << " ";
  std::cout << std::endl;
}


/* Functions from QuEST_gpu_local.cu */

#ifdef __cplusplus
extern "C" {
#endif

qreal statevec_getRealAmpLocal(Qureg qureg, long long int index);
qreal statevec_getImagAmpLocal(Qureg qureg, long long int index);

Complex statevec_calcInnerProductLocal(Qureg bra, Qureg ket);
void statevec_compactUnitaryLocal(Qureg qureg, const int targetQubit, Complex alpha, Complex beta);
// statevec_unitaryLocal
void statevec_unitaryLocal(Qureg qureg, const int targetQubit, ComplexMatrix2 u);
void statevec_controlledCompactUnitaryLocal(Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta);
// statevec_controlledUnitaryLocal
void statevec_controlledUnitaryLocal(Qureg qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u);
// statevec_multiControlledUnitaryLocal
void statevec_multiControlledUnitaryLocal(Qureg qureg, long long int ctrlQubitsMask, long long int ctrlFlipMask, const int targetQubit, ComplexMatrix2 u);
void statevec_pauliXLocal(Qureg qureg, const int targetQubit);
void statevec_controlledNotLocal(Qureg qureg, const int controlQubit, const int targetQubit);
void statevec_pauliYLocal(Qureg qureg, const int targetQubit);
void statevec_pauliYConjLocal(Qureg qureg, const int targetQubit);
void statevec_controlledPauliYLocal(Qureg qureg, const int controlQubit, const int targetQubit);
void statevec_controlledPauliYConjLocal(Qureg qureg, const int controlQubit, const int targetQubit);
void statevec_hadamardLocal(Qureg qureg, const int targetQubit);
qreal statevec_findProbabilityOfZeroLocal(Qureg qureg, const int measureQubit);

// TODO
void statevec_collapseToKnownProbOutcomeLocal(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb);
void statevec_swapQubitAmpsLocal(Qureg qureg, int qb1, int qb2);
void statevec_multiControlledTwoQubitUnitaryLocal(Qureg qureg, long long int ctrlMask, const int q1, const int q2, ComplexMatrix4 u);
void statevec_multiControlledMultiQubitUnitaryLocal(Qureg qureg, long long int ctrlMask, int* targs, const int numTargs, ComplexMatrixN u);


// densmatr
void densmatr_initPureState(Qureg targetQureg, Qureg copyQureg);
void densmatr_initPlusState(Qureg qureg);
void densmatr_initClassicalState(Qureg qureg, long long int stateInd);
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb);
void densmatr_mixDensityMatrix(Qureg combineQureg, qreal otherProb, Qureg otherQureg);
void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit, qreal dephFac);
void densmatr_mixDephasing(Qureg qureg, const int targetQubit, qreal dephase);
void densmatr_mixTwoQubitDephasing(Qureg qureg, int qubit1, int qubit2, qreal dephase);
void densmatr_mixDepolarising(Qureg qureg, const int targetQubit, qreal depolLevel);
void densmatr_mixDamping(Qureg qureg, const int targetQubit, qreal damping);
void densmatr_mixTwoQubitDepolarising(Qureg qureg, int qubit1, int qubit2, qreal depolLevel);
qreal densmatr_calcFidelity(Qureg qureg, Qureg pureState);
qreal densmatr_calcHilbertSchmidtDistanceSquared(Qureg a, Qureg b);
qreal densmatr_calcPurity(Qureg qureg);
qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome);
qreal densmatr_findProbabilityOfZero(Qureg qureg, const int measureQubit);
qreal densmatr_calcTotalProb(Qureg qureg);
qreal densmatr_calcHilbertSchmidtDistance(Qureg a, Qureg b);
qreal densmatr_calcInnerProduct(Qureg a, Qureg b);


#ifdef __cplusplus
}
#endif

// TODO
void statevec_unitaryDistributed (Qureg qureg,
        Complex rot1, Complex rot2,
        ComplexArray stateVecUp,
        ComplexArray stateVecLo,
        ComplexArray stateVecOut);
void statevec_controlledUnitaryDistributed (Qureg qureg, const int controlQubit,
        Complex rot1, Complex rot2,
        ComplexArray stateVecUp,
        ComplexArray stateVecLo,
        ComplexArray stateVecOut);
void statevec_multiControlledUnitaryDistributed (
        Qureg qureg, 
        const int targetQubit, 
        long long int ctrlQubitsMask, long long int ctrlFlipMask,
        Complex rot1, Complex rot2,
        ComplexArray stateVecUp,
        ComplexArray stateVecLo,
        ComplexArray stateVecOut);
void statevec_collapseToKnownProbOutcomeDistributedRenorm (Qureg qureg, const int measureQubit, const qreal totalProbability);
void statevec_swapQubitAmpsDistributed(Qureg qureg, int pairRank, int qb1, int qb2);
void statevec_collapseToOutcomeDistributedSetZero(Qureg qureg);

#define DEFAULT_THREADS_PER_BLOCK 1024

#endif
