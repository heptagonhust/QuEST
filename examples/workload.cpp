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

    Qureg q = createQureg(32, Env);

    float q_measure[30];
    tGate(q, 25);
    controlledNot(q, 28, 21);
    controlledRotateX(q, 17, 5, 0.3293660327520663);
    rotateX(q, 27, 3.9207427542347926);
    tGate(q, 3);
    controlledRotateZ(q, 27, 19, 5.459935259485407);
    controlledRotateX(q, 26, 3, 4.736990305652013);
    controlledRotateZ(q, 8, 11, 3.594728080156504);
    rotateX(q, 10, 4.734238389048838);
    rotateY(q, 8, 4.959946047271496);
    rotateZ(q, 5, 1.0427019597472071);
    controlledRotateZ(q, 27, 0, 5.971846444908922);
    pauliZ(q, 0);
    tGate(q, 4);
    controlledRotateX(q, 29, 17, 5.885491371058282);
    tGate(q, 6);
    tGate(q, 23);
    controlledRotateZ(q, 28, 11, 4.12817017175927);
    hadamard(q, 17);
    controlledNot(q, 17, 3);
    rotateZ(q, 0, 3.8932024879670144);
    controlledRotateY(q, 22, 28, 5.384534074265311);
    controlledNot(q, 29, 5);
    sGate(q, 8);
    controlledPauliY(q, 23, 14);
    controlledPauliY(q, 18, 17);
    controlledPauliY(q, 9, 15);
    pauliY(q, 6);
    controlledNot(q, 19, 29);
    controlledPauliY(q, 1, 25);
    pauliZ(q, 8);
    pauliY(q, 2);
    pauliX(q, 9);
    controlledPauliY(q, 4, 12);
    controlledRotateY(q, 17, 14, 4.308551563407819);
    rotateX(q, 11, 5.512541996174936);
    pauliX(q, 24);
    pauliY(q, 7);
    tGate(q, 18);
    hadamard(q, 27);
    pauliZ(q, 29);
    controlledNot(q, 15, 13);
    controlledRotateZ(q, 16, 3, 1.375780109278382);
    pauliZ(q, 28);
    controlledRotateX(q, 23, 20, 4.5063180242513505);
    pauliZ(q, 21);
    sGate(q, 6);
    rotateX(q, 18, 2.337847412996701);
    tGate(q, 21);
    rotateY(q, 21, 2.5090791677412008);
    controlledRotateY(q, 13, 8, 2.5004731956129143);
    controlledPauliY(q, 2, 10);
    controlledNot(q, 22, 1);
    pauliX(q, 3);
    pauliX(q, 3);
    rotateX(q, 29, 2.723779839507815);
    hadamard(q, 2);
    controlledRotateZ(q, 29, 8, 3.7348369237156893);
    controlledPauliY(q, 17, 10);
    pauliY(q, 13);
    sGate(q, 13);
    controlledRotateY(q, 21, 4, 1.0006054937741466);
    tGate(q, 12);
    controlledPauliY(q, 14, 4);
    pauliZ(q, 11);
    controlledPauliY(q, 13, 4);
    controlledNot(q, 18, 4);
    rotateX(q, 27, 5.456179791071725);
    rotateY(q, 23, 2.3597295726584417);
    pauliY(q, 18);
    rotateX(q, 20, 4.663082879319556);
    controlledRotateY(q, 17, 3, 3.379870011915129);
    pauliZ(q, 17);
    controlledRotateY(q, 27, 8, 4.729823556797339);
    rotateY(q, 10, 1.9665821442518263);
    hadamard(q, 21);
    hadamard(q, 23);
    pauliY(q, 1);
    hadamard(q, 20);
    pauliX(q, 19);
    rotateZ(q, 14, 2.0069208879155003);
    sGate(q, 17);
    rotateY(q, 7, 1.1987039711422482);
    controlledRotateY(q, 16, 25, 5.525016274711897);
    pauliZ(q, 2);
    pauliY(q, 19);
    controlledRotateX(q, 5, 22, 5.474489446026321);
    controlledRotateZ(q, 22, 25, 2.054682600662274);
    controlledPauliY(q, 19, 6);
    tGate(q, 14);
    rotateY(q, 25, 5.689131875569378);
    rotateY(q, 29, 5.261268123984145);
    rotateY(q, 18, 5.340898512406205);
    controlledRotateY(q, 5, 8, 0.2087337909838518);
    tGate(q, 7);
    pauliY(q, 2);
    controlledNot(q, 26, 12);
    controlledRotateX(q, 27, 15, 5.113996985399576);
    hadamard(q, 20);
    pauliZ(q, 8);
    tGate(q, 10);
    hadamard(q, 9);
    pauliZ(q, 8);
    rotateY(q, 21, 5.899576921821051);
    pauliY(q, 24);
    controlledRotateZ(q, 11, 23, 1.1916005322627135);
    controlledRotateZ(q, 18, 7, 2.558871283621717);
    pauliX(q, 15);
    hadamard(q, 23);
    rotateX(q, 10, 5.259645311585795);
    controlledNot(q, 19, 26);
    rotateY(q, 18, 5.982090815955244);
    controlledRotateX(q, 10, 26, 1.9709969724073322);
    tGate(q, 22);
    hadamard(q, 20);
    controlledRotateX(q, 6, 12, 0.8115870637427451);
    controlledRotateZ(q, 22, 7, 2.0426711293536624);
    hadamard(q, 22);
    rotateY(q, 17, 5.853295168431142);
    controlledPauliY(q, 12, 22);
    rotateZ(q, 7, 2.8526004701547407);
    controlledRotateY(q, 25, 23, 2.558389682561494);

    printf("\n");
    for(long long int i=0; i<30; ++i){
        q_measure[i] = calcProbOfOutcome(q,  i, 1);
        //printf("  probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
        fprintf(fp, "Probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
    }
    fprintf(fp, "\n");
    printf("\n");


    for(int i=0; i<10; ++i){
        Complex amp = getAmp(q, i);
        //printf("Amplitude of %dth state vector: %f\n", i, prob);
	fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }

    double t2 = get_wall_time();
    printf("Complete the simulation takes time %12.6f seconds.", t2 - t1);
    printf("\n");
    destroyQureg(q, Env);
    destroyQuESTEnv(Env);

    return 0;
}
