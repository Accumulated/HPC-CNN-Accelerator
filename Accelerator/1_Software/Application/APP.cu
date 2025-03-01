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
#include <vector>

int main(void){


    Matrix *Input = new Matrix(
        112, 127, 16, MBConv6_1_expansion_conv_conv2d_input_matrix, DefineOnDevice
    );

    Dimension InputDim = Dimension{112, 127, 16};
    Dimension* M = &InputDim;

    // Create a vector of MBConv objects
    std::vector<MBConv*> MBConvLayers;

    // Initialize the MBConv objects and add them to the vector
    for (int i = 1; i <= 15; ++i) {
      MBConv* layer = nullptr;
      switch (i) {
          case 1: layer = new MBConv(&MBConv6_1_Layers, M); break;
          case 2: layer = new MBConv(&MBConv6_2_Layers, M); break;
          case 3: layer = new MBConv(&MBConv6_3_Layers, M); break;
          case 4: layer = new MBConv(&MBConv6_4_Layers, M); break;
          case 5: layer = new MBConv(&MBConv6_5_Layers, M); break;
          case 6: layer = new MBConv(&MBConv6_6_Layers, M); break;
          case 7: layer = new MBConv(&MBConv6_7_Layers, M); break;
          case 8: layer = new MBConv(&MBConv6_8_Layers, M); break;
          case 9: layer = new MBConv(&MBConv6_9_Layers, M); break;
          case 10: layer = new MBConv(&MBConv6_10_Layers, M); break;
          case 11: layer = new MBConv(&MBConv6_11_Layers, M); break;
          case 12: layer = new MBConv(&MBConv6_12_Layers, M); break;
          case 13: layer = new MBConv(&MBConv6_13_Layers, M); break;
          case 14: layer = new MBConv(&MBConv6_14_Layers, M); break;
          case 15: layer = new MBConv(&MBConv6_15_Layers, M); break;
      }
      MBConvLayers.push_back(layer);
      M = layer->MBConv_GetOutputDim();
  }

  // Process the input through the MBConv layers
  Matrix* output = Input;
  for (auto& layer : MBConvLayers) {
    output = (*layer)(output);
    output->Matrix_DumpDeviceMemory();
  }

}