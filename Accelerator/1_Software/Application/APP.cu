#include "CommonInclude.h"
#include "MBConv6_1_Abstraction.h"

int main(void){

    Dimension InputDim = Dimension{224, 254, 16};

    MBConv* MBConv6_1 = new MBConv(
        &MBConv6_1_Layers,
        &InputDim
    );

    Dimension* M = MBConv6_1 -> MBConv_GetOutputDim();

}