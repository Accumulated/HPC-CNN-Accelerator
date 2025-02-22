#pragma once

class ReductionSum{

public:

    Matrix *Output;

    /* This is used for large arrays only if needed */
    Matrix *TransitionMatrix;

    ReductionSum();
    ~ReductionSum();

    Matrix* operator()(Matrix* D_Input);

};