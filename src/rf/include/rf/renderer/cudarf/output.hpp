#ifndef __CUDARF_OUTPUT
#define __CUDARF_OUTPUT

#include <rf/renderer/cudarf/memory.hpp>

namespace cudarf
{
struct CudaOutput {
    int dev_id;
    int SMPCount;
    int clockRate;

    memory::DeviceBuffer<uchar4> devOutput;
    memory::PinnedBuffer<unsigned char> cpuOutput;

    int width;
    int height;

    struct {
        cudaGraphicsResource *cuda_pbo_resource;
        unsigned int fbo_tex;
        unsigned int fbo;
        unsigned int depth_renderbuffer;
        unsigned int pbo;
    } gl_output;

    CudaOutput() = delete;
    CudaOutput(const CudaOutput&) = delete;
    CudaOutput &operator=(const CudaOutput &) = delete;

    CudaOutput(int width, int height);
    ~CudaOutput() = default;

    // copy 'cpu_output' to gl_output.fbo_tex
    void present();
};

}

#endif
