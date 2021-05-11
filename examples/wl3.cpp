//total number of qubit: 30
//total number of qubit operatations: 667
//estimated time: 3783.9266747315614 second.
#include "QuEST.h"
#include "stdio.h"
#include "mytimer.hpp"


int main (int narg, char *argv[]) {

    QuESTEnv Env = createQuESTEnv();
    double t1 = get_wall_time();

    int NUM=32;
    Qureg q = createQureg(NUM, Env);

    puts("statevec_rotateX statevec_compactUnitaryLocal");
    for(int i=1;i<100;i++) {
        if(i%NUM==0) printf("%d over\n",i); 
        if (i%NUM == 0 ) continue;
        rotateX(q, i%NUM - 1, 3.13624578345); //statevec_rotateX statevec_compactUnitaryLocal 
    }
    double t2 = get_wall_time();
    printf("Complete the simulation takes time %12.6f seconds.", t2 - t1);

    // puts("statevec_rotateX statevec_compactUnitaryLocal");
    // for(int i=0;i<900;i++) {
    //     if(i%30==0) printf("%d over\n",i);
    //     if (i/30==29) continue;
    //     rotateX(q, i/30, 3.13624578345); //statevec_rotateX statevec_compactUnitaryLocal 
    // }
    t1 = get_wall_time();
    printf("Complete the simulation takes time %12.6f seconds.", t1 - t2);
    // calcProbOfOutcome(q,  0, 1);
    destroyQureg(q, Env);
    destroyQuESTEnv(Env);

    return 0;
}