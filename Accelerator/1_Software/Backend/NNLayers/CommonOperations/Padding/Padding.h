#pragma once



class Padding{


public:

    Matrix *Output;

    Padding();
    ~Padding();

    Matrix* operator()(Matrix* D_Input, int Boundary); // Corrected the typo here

};