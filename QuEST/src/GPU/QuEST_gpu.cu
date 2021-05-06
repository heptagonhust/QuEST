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

// TODO
// void statevec_unitaryDistributed (Qureg qureg,
//         Complex rot1, Complex rot2,
//         ComplexArray stateVecUp,
//         ComplexArray stateVecLo,
//         ComplexArray stateVecOut){}
// void statevec_controlledUnitaryDistributed (Qureg qureg, const int controlQubit,
//         Complex rot1, Complex rot2,
//         ComplexArray stateVecUp,
//         ComplexArray stateVecLo,
//         ComplexArray stateVecOut){}
void statevec_multiControlledUnitaryDistributed (
        Qureg qureg, 
        const int targetQubit, 
        long long int ctrlQubitsMask, long long int ctrlFlipMask,
        Complex rot1, Complex rot2,
        ComplexArray stateVecUp,
        ComplexArray stateVecLo,
        ComplexArray stateVecOut){}
void statevec_collapseToKnownProbOutcomeDistributedRenorm (Qureg qureg, const int measureQubit, const qreal totalProbability){}
void statevec_swapQubitAmpsDistributed(Qureg qureg, int pairRank, int qb1, int qb2){}
void statevec_collapseToOutcomeDistributedSetZero(Qureg qureg){}