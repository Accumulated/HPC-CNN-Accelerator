#include "CommonInclude.h"
#include "MBConv6_1_Abstraction.h"

int main(void){

    Dimension InputDim = Dimension{200, 200, 3};

    MBConv* MBConv6_1 = new MBConv(
        &MBConv6_1_Layers,
        &InputDim
    );

}