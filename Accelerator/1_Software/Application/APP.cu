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
#include "Input_Matrix.h"

int main(void){

    Matrix *Input = new Matrix(
        112, 127, 16, MBConv6_1_expansion_conv_conv2d_input_matrix, DefineOnDevice
    );

    Dimension InputDim = Dimension{112, 127, 16};
    Dimension* M = &InputDim;

    MBConv* MBConv6_1 = new MBConv(
        &MBConv6_1_Layers,
        M
    );
    M = MBConv6_1 -> MBConv_GetOutputDim();

    Matrix* output = (*MBConv6_1)(Input);

    output -> Matrix_DumpDeviceMemory();

}