#include "CommonInclude.h"
#include "Input_For_Stem_Layer.h"
#include "Stem_Abstraction.h"
#include "MBConv1_0_Abstraction.h"
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
#include "HLayer_Abstraction.h"
#include <vector>

int main(void){

    Matrix *Input = new Matrix(
      224, 254, 3, Input_for_stem_conv, DefineOnDevice
    );
    Dimension InputDim = Dimension{224, 254, 3};

    // Keep track of my moving dimension across the layers
    Dimension* M = &InputDim;

    // Create a vector of any layer defined
    std::vector<IBasicLayer*> NNModel;

    SLayer* Startlayer = new SLayer(&StemLayer, M);
    M = Startlayer -> SLayer_GetOutputDim();

    NNModel.push_back(
      Startlayer
    );

    // Initialize the MBConv objects and add them to the vector
    for (int i = 0; i <= 15; ++i) {
      MBConv* layer = nullptr;
      switch (i) {
          case 0: layer = new MBConv(&MBConv1_0_Layers, M); break;
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
      NNModel.push_back(layer);
      M = layer->MBConv_GetOutputDim();
  }

  NNModel.push_back(
    new HLayer(&HeadLayer, M)
  );

  // Process the input through the MBConv layers
  Matrix* output = Input;
  for (auto& layer : NNModel) {
    output = (*layer)(output);
  }
  output->Matrix_DumpDeviceMemory();

}