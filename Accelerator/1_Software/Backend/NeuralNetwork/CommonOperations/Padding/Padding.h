#pragma once


class PaddingLayer{

public:

    Matrix *Output;
    int Boundary;


    PaddingLayer(Dimension *InputDim, int Boundary);
    ~PaddingLayer();

    Matrix* operator()(Matrix* D_Input);

};