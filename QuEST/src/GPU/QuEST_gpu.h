#ifndef _QUEST_GPU_H_
#define _QUEST_GPU_H_

/*
* Bit twiddling functions are defined seperately here in the CPU backend, 
* since the GPU backend  needs a device-specific redefinition to be callable 
* from GPU kernels. These are called in both QuEST_cpu and QuEST_cpu_distributed 
* and defined in here since public inline methods in C must go in the header
*/

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


#include <assert.h>

#endif
