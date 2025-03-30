#pragma once

class IBasicLayer {

public:

    // Virtual destructor to ensure proper cleanup of derived classes
    // cleans up streams
    virtual ~IBasicLayer() {
        if (streams) {

            for (int i = 0; i < numberOfStreams; ++i) {
                cudaStreamDestroy(streams[i]);  // Destroy all streams
            }
            delete[] streams;

            streams = nullptr;
        }
    }


    virtual Matrix** operator()(Matrix** D_input) = 0;


protected:
    static const int numberOfStreams = 5;
    static cudaStream_t* streams;         

    // Constructor initializes streams
    IBasicLayer() {
        if (!streams) {
            streams = new cudaStream_t[numberOfStreams];
            for (int i = 0; i < numberOfStreams; ++i) {
                cudaError_t err = cudaStreamCreate(&streams[i]);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to create CUDA stream");
                }
            }
        }
    }
};

