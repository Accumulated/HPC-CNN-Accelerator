#include "CommonInclude.h"
#include "MBConv6_1_Abstraction.h"
#include "MBConv6_2Abstraction.h"
#include "MBConv6_3Abstraction.h"
#include "MBConv6_4Abstraction.h"
#include "MBConv6_5Abstraction.h"
#include "MBConv6_6Abstraction.h"
#include "MBConv6_7Abstraction.h"
#include "MBConv6_8Abstraction.h"
#include "MBConv6_9Abstraction.h"
#include "MBConv6_10Abstraction.h"
#include "MBConv6_11Abstraction.h"
#include "MBConv6_12Abstraction.h"
#include "MBConv6_13Abstraction.h"
#include "MBConv6_14Abstraction.h"
#include "MBConv6_15Abstraction.h"

int main(void){

    Dimension InputDim = Dimension{112, 127, 16};
    Dimension* M = &InputDim;

    MBConv* MBConv6_1 = new MBConv(
        &MBConv6_1_Layers,
        M
    );
    M = MBConv6_1 -> MBConv_GetOutputDim();


    MBConv* MBConv6_2 = new MBConv(
        &MBConv6_2_Layers,
        M
    );
    M = MBConv6_2 -> MBConv_GetOutputDim();


    MBConv* MBConv6_3 = new MBConv(
        &MBConv6_3_Layers,
        M
    );
    M = MBConv6_3 -> MBConv_GetOutputDim();

    MBConv* MBConv6_4 = new MBConv(
        &MBConv6_4_Layers,
        M
    );
    M = MBConv6_4 -> MBConv_GetOutputDim();

    MBConv* MBConv6_5 = new MBConv(
        &MBConv6_5_Layers,
        M
    );

    M = MBConv6_5 -> MBConv_GetOutputDim();

}