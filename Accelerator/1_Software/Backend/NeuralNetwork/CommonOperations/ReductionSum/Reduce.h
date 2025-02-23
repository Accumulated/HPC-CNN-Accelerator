#pragma once

class ReductionSum: public IBasicLayer{

public:

    Matrix *Output;

    /* This is used for large arrays only if needed */
    Matrix *TransitionMatrix;

    ReductionSum(Dimension* InputDim);
    ~ReductionSum();

    Dimension* InputDim;
    Dimension OutputDim;

    Matrix* operator()(Matrix* D_Input);
    void ResetDims();

};