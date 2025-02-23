#pragma once


typedef struct MBConv_SqueezeExcite{

    Basiclayer SQ1;
    Basiclayer SQ2;

}MBConv_SqueezeExcite;


typedef struct MBConv_Abstraction{


    Basiclayer Expansion;
    Basiclayer DepthWise;
    Basiclayer Project;
    MBConv_SqueezeExcite SQueezeExcite;

    unsigned int Stride;
    unsigned int Padding;
    SkipID Skip;

}MBConv_Abstraction;



class MBConv: public IBasicLayer{


public:

    MBConv(const MBConv_Abstraction* MBConvDetails);

    ~MBConv();

    Matrix* operator()(Matrix* D_input);
    Matrix* MBConv_SKIPID(Matrix *D_Input);


    std::vector<IBasicLayer*> layers;


    const MBConv_Abstraction *MBConvDetails = nullptr;
    Matrix* Output;

};