#include <stdio.h>
#include "QuEST.h"
#include "mytimer.hpp"

/*
    Simulate the 6x6 two-dimentional Qubits grids with the 3th qubit broken.
    The total simulated qubits is 35 and the circuit depth is 20.
*/

int main () {

    QuESTEnv env = createQuESTEnv();

    FILE *fp, *fvec;
    if(env.rank==0){

        fp=fopen("probs.dat", "w");
        if(fp==NULL){
            printf("    open probs.dat failed, Bye!");
            return 0;
        }

        fvec=fopen("stateVectors.dat", "w");
        if(fvec==NULL){
            printf("    open stateVector.dat failed, Bye!");
            return 0;
        }
    }

    int numQubits = 35;
    
    Qureg QReg = createQureg(numQubits, env);
    initZeroState(QReg);

    /* start timing */
    double t0 = get_wall_time();

    #include"circuit.dat"

    qreal prob;
    for(int ind=0; ind<numQubits; ++ind){
        prob = calcProbOfOutcome(QReg, ind, 1);
        if(env.rank==0){
            printf("Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
            fprintf(fp, "Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
        }
    }

    for(int i=0; i<10; ++i){
        Complex amp = getAmp(QReg, i);
	    if(env.rank==0) fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }

    /* finish timing */
    double t = get_wall_time() - t0;
    if(env.rank==0) printf("Time cost: %lf\n", t);

    destroyQureg(QReg, env);
    destroyQuESTEnv(env);

    return 0;
}
