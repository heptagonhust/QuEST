//total number of qubit: 30
//total number of qubit operatations: 667
//estimated time: 3783.9266747315614 second.
#include "QuEST.h"
#include "stdio.h"
#include "mytimer.hpp"


int main (int narg, char *argv[]) {

    QuESTEnv Env = createQuESTEnv();
    double t1 = get_wall_time();

    FILE *fp=fopen("probs.dat", "w");
    if(fp==NULL){
        printf("    open probs.dat failed, Bye!");
        return 0;
    }

    FILE *fvec=fopen("stateVector.dat", "w");
    if(fp==NULL){
        printf("    open stateVector.dat failed, Bye!");
        return 0;
    }

    int N = 3;
    Qureg q = createQureg(N, Env);

    float q_measure[N];
    
    // Single gate that make pure zero state to non-zero here causes bugs, 
    // including hadamard, rotateX/Y, pauliX/Y
    // excluding rotateZ, pauliZ, tGate, sGate 
    //hadamard(q, N - 1);
    hadamard(q, 2);

    // Any relaying qubit will make the things worse
    // controlledPauliY(q, 17, 29);
    // rotateX(q, 21, 5.370693097083298); // make qureg not zero, it can be commented
    // controlledPauliY(q, 21, 25); 

    // Controlled gate cause the bugs,
    // including controlledNot, controlledRotateX/Y, controlledPauliY,
    // excluding controlledRotateZ
    // controlledPauliY(q, 29, 25);
    controlledNot(q, N - 1, 1);
    // control > target => error
    // control < target => fine (GHZ_QFT)


    printf("\n");
    for(long long int i=0; i<N; ++i){
        q_measure[i] = calcProbOfOutcome(q,  i, 1);
        //printf("  probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
        fprintf(fp, "Probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
    }
    fprintf(fp, "\n");
    printf("\n");


    // for(int i=0; i<5; ++i){
    //     Complex amp = getAmp(q, i);
    //     //printf("Amplitude of %dth state vector: %f\n", i, prob);
	// fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    // }

    double t2 = get_wall_time();
    printf("Complete the simulation takes time %12.6f seconds.", t2 - t1);
    printf("\n");
    destroyQureg(q, Env);
    destroyQuESTEnv(Env);

    return 0;
}
