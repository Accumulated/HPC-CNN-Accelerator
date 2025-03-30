#include "CommonInclude.h"
#include "IBasicLayer.h"

// Initialize the streams to a nullptr, the first instance calling the 
// IBasicLayer constructor is going to initialize the streams - making sure
// that only 5 or 10 or n streams shared across all the layers. Otherwise, 
// this will have over 10000 streams running everywhere (very bad)
cudaStream_t* IBasicLayer::streams = nullptr;
