#include "EfficientNet.h"
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




EfficientNet::EfficientNet() {

    Input = new Matrix*[this -> numberOfStreams]; // Allocate an array of Matrix pointers

    for (int i = 0; i < this -> numberOfStreams; i++) {

        Input[i] = new Matrix(224, 254, 3, Input_for_stem_conv, DefineOnDevice);

    }

    InputDim = Dimension{224, 254, 3};
    M = &InputDim;

    SLayer* Startlayer = new SLayer(&StemLayer, M);
    M = Startlayer->SLayer_GetOutputDim();
    NNModel.push_back(Startlayer);

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

    NNModel.push_back(new HLayer(&HeadLayer, M));
}

EfficientNet::~EfficientNet() {
    // Clean up resources
    for (auto layer : NNModel) {
        delete layer;
    }
    delete Input;
}

Matrix** EfficientNet:: operator()(Matrix **D_input){

    return (Matrix**) nullptr;
}

Matrix** EfficientNet:: operator()(){

    Matrix** output = Input;
    for (auto& layer : NNModel) {
        output = (*layer)(output);
    }

    output[0] -> Matrix_DumpDeviceMemory();

    return output;
}

