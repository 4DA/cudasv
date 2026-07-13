#include <string>

#include <spdlog/spdlog.h>

#include <cuda_gl_interop.h>

#include <rf/renderer/cudarf/types.hpp>
#include <rf/renderer/cudarf/output.hpp>

#include "glcommon.hpp"
#include "helpers.hpp"

#define CHECK_GL_ERROR() {\
    GLenum error = glGetError();\
    if(error != GL_NO_ERROR) {\
        printf("ogl error %s(%d): %d\n", __FILE__, __LINE__, error);\
        exit(-1);\
    } }

/* ...vertex shader */
static const char vshader[] = {
    "precision lowp float;\n"
    "attribute vec2 XY;\n"
    "attribute vec2 UV;\n"
    "varying vec2 uv;\n"
    "void main(void)\n"
    "{\n"
    "   uv = UV;\n"
    "   gl_Position = vec4(XY, 0.0, 1.0);\n"
    "}\n"
};

static const char fshader[] = {
    "precision lowp float;\n"
    "uniform sampler2D tex;\n"
    "varying vec2 uv;\n"
    "void main(void)\n"
    "{\n"
    "    gl_FragColor = texture2D(tex, uv);\n"
    "}\n"
};

static GLuint shaderID = 0;
static GLuint texLoc = 0;
static GLuint xyLoc = 0;
static GLuint uvLoc = 0;

static void print_dev_prop(const cudaDeviceProp &devProp,
                           int clockRate,
                           int asyncEngineCount,
                           int kernelExecTimeout)
{
    SPDLOG_INFO("Major revision number:         {}", devProp.major);
    SPDLOG_INFO("Minor revision number:         {}", devProp.minor);
    SPDLOG_INFO("Name:                          {}", devProp.name);
    SPDLOG_INFO("Total global memory:           {}", devProp.totalGlobalMem);
    SPDLOG_INFO("Total shared memory per block: {}", devProp.sharedMemPerBlock);
    SPDLOG_INFO("Total registers per block:     {}", devProp.regsPerBlock);
    SPDLOG_INFO("Warp size:                     {}", devProp.warpSize);
    SPDLOG_INFO("Maximum memory pitch:          {}", devProp.memPitch);
    SPDLOG_INFO("Maximum threads per block:     {}", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) {
        SPDLOG_INFO("Maximum dimension {} of block:  {}", i, devProp.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i) {
        SPDLOG_INFO("Maximum dimension {} of grid:   {}", i, devProp.maxGridSize[i]);
    }
    SPDLOG_INFO("Clock rate:                    {}", clockRate);
    SPDLOG_INFO("Total constant memory:         {}", devProp.totalConstMem);
    SPDLOG_INFO("Texture alignment:             {}", devProp.textureAlignment);
    SPDLOG_INFO("Concurrent copy and execution: {}", asyncEngineCount > 0 ? "Yes" : "No");
    SPDLOG_INFO("Number of multiprocessors:     {}", devProp.multiProcessorCount);
    SPDLOG_INFO("Kernel execution timeout:      {}", kernelExecTimeout ? "Yes" : "No");
}

static inline const char* _ConvertSMVer2ArchName(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the GPU Arch name)
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"},
        {0x32, "Kepler"},
        {0x35, "Kepler"},
        {0x37, "Kepler"},
        {0x50, "Maxwell"},
        {0x52, "Maxwell"},
        {0x53, "Maxwell"},
        {0x60, "Pascal"},
        {0x61, "Pascal"},
        {0x62, "Pascal"},
        {0x70, "Volta"},
        {0x72, "Xavier"},      // Volta-based Tegra (GV10B)
        {0x75, "Turing"},
        {0x80, "Ampere"},      // A100, etc.
        {0x86, "Ampere"},      // GA10x (desktop/mobile)
        {0x87, "Ampere"},      // Orin (Jetson) – SM 8.7
        {0x89, "Ada Lovelace"},// RTX 40-series – SM 8.9
        {0x90, "Hopper"},      // H100 – SM 9.0
        {0xA0, "Blackwell"},   // B200/RTX Blackwell – SM 10.0
        {-1,   "Graphics Device"}
    };

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    SPDLOG_INFO("MapSMtoArchName for SM {}.{} is undefined. Default to use {}",
                major, minor, nGpuArchNameSM[index - 1].name);

    return nGpuArchNameSM[index - 1].name;
}
  // end of GPU Architecture definitions

static void shader_debug(GLuint obj, GLenum status, const char *op)
{
    auto get_info_log = [&]() -> std::string {
        int len = 0;
        if (status == GL_COMPILE_STATUS)
            glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &len);
        else
            glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
        if (len <= 0) return {};
        std::string buf(len, '\0');
        if (status == GL_COMPILE_STATUS)
            glGetShaderInfoLog(obj, len, nullptr, buf.data());
        else
            glGetProgramInfoLog(obj, len, nullptr, buf.data());
        return buf;
    };

    std::string log = get_info_log();
    if (!log.empty())
        SPDLOG_ERROR("--- {} log ---\n{}", op, log);

    int success = 0;
    if (status == GL_COMPILE_STATUS) {
        glGetShaderiv(obj, status, &success);
        if (!success) {
            int len = 0;
            glGetShaderiv(obj, GL_SHADER_SOURCE_LENGTH, &len);
            if (len > 0) {
                std::string src(len, '\0');
                glGetShaderSource(obj, len, nullptr, src.data());
                if (!src.empty())
                    SPDLOG_ERROR("--- {} code ---\n{}", op, src);
            }
        }
    } else {
        glGetProgramiv(obj, status, &success);
    }

    if (!success) {
        SPDLOG_ERROR("--- {} failed ---", op);
        exit(-1);
    }
}

// Takes shader source strings, compiles them, and builds a shader program
static unsigned int load_shader_src_strings(const char* vertSrc,
                                            int vertSrcSize,
                                            const char* fragSrc,
                                            int fragSrcSize,
                                            unsigned char link,
                                            unsigned char debugging)
{
    GLuint prog = 0;
    GLuint vertShader;
    GLuint fragShader;

    // Create the program
    prog = glCreateProgram();

    // Create the GL shader objects
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Load shader sources into GL and compile
    glShaderSource(vertShader, 1, (const char**)&vertSrc, &vertSrcSize);
    glCompileShader(vertShader);

    if (debugging) {shader_debug(vertShader, GL_COMPILE_STATUS, "Vert Compile");}
    glShaderSource(fragShader, 1, (const char**)&fragSrc, &fragSrcSize);
    glCompileShader(fragShader);

    if (debugging) {shader_debug(fragShader, GL_COMPILE_STATUS, "Frag Compile");}

    // Attach the shaders to the program
    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);

    // Delete the shaders
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // Link (if requested) and validate the shader program
    if (link) {
        glLinkProgram(prog);
        if (debugging) {
            shader_debug(prog, GL_LINK_STATUS, "Program Link");
        }
        glValidateProgram(prog);
        if (debugging) {
            shader_debug(prog, GL_VALIDATE_STATUS, "Program Validate");
        }
    }

    return prog;
}

static void init_shader(void)
{
    static bool initialized = false;

    if (initialized)
        return;

    GLuint id;
    id = load_shader_src_strings(vshader,
                                 sizeof(vshader),
                                 fshader,
                                 sizeof(fshader),
                                 GL_TRUE,
                                 GL_TRUE);

    if(!id) {
        printf("failed to create shader program\n");
        exit(-1);
    }

    CHECK_GL_ERROR();

    shaderID = id;
    xyLoc = glGetAttribLocation(shaderID, "XY");
    uvLoc = glGetAttribLocation(shaderID, "UV");
    texLoc = glGetUniformLocation(shaderID, "tex");

    initialized = true;
}

static void draw_quad(GLuint tex, int i)
{
    init_shader();

    static const GLfloat    verts[] = {
        -1,    1,
         1,    1,
        -1,   -1,
        -1,   -1,
         1,    1,
         1,   -1,
    };

    static const GLfloat    texcoords[] = {
        0,  0,
        1,  0,
        0,  1,
        0,  1,
        1,  0,
        1,  1,
    };

    glUseProgram(shaderID);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(texLoc, 0);

    glVertexAttribPointer(xyLoc, 2, GL_FLOAT, GL_FALSE, 0, verts);
    glEnableVertexAttribArray(xyLoc);

    glVertexAttribPointer(uvLoc, 2, GL_FLOAT, GL_FALSE, 0, texcoords);
    glEnableVertexAttribArray(uvLoc);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glDisableVertexAttribArray(xyLoc);
    glDisableVertexAttribArray(uvLoc);

    glUseProgram(0);
}

void cudarf::CudaOutput::present(const uchar4* devBuffer)
{
    // TODO: research if cuda -> opengl memcpy possible

    CUDA_CHK(cudaMemcpy(cpu_output, devBuffer, width * height * 4, cudaMemcpyDeviceToHost));

    glBindTexture(GL_TEXTURE_2D, gl_output.fbo_tex);

    glTexImage2D(GL_TEXTURE_2D,
                 0,                // GLint level,
                 GL_RGBA,          // GLint internalformat,
                 width,
                 height,
                 0,                // GLint border,
                 GL_RGBA,          // GLenum format,
                 GL_UNSIGNED_BYTE, // GLenum type,
                 cpu_output);      // const void * data);

    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_DEPTH_TEST);
    draw_quad(gl_output.fbo_tex, 0);
    glEnable(GL_DEPTH_TEST);
}

cudarf::CudaOutput::CudaOutput(int width, int height):
    width(width),
    height(height)
{
    int dev_id = 0;

    int count;
    cudaDeviceProp prop;
    CUDA_CHK(cudaGetDeviceCount(&count));

    SPDLOG_INFO("Found cuda devices: {}", count);

    if (count == 0) {
        assert(false);
    }

    CUDA_CHK(cudaGetDeviceProperties(&prop, dev_id));

    int computeMode = -1;
    int major = 0;
    int minor = 0;
    int asyncEngineCount = 0;
    int kernelExecTimeout = 0;
    CUDA_CHK(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev_id));
    CUDA_CHK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id));
    CUDA_CHK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));
    CUDA_CHK(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev_id));
    CUDA_CHK(cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, dev_id));
    CUDA_CHK(cudaDeviceGetAttribute(&kernelExecTimeout, cudaDevAttrKernelExecTimeout, dev_id));

    print_dev_prop(prop, clockRate, asyncEngineCount, kernelExecTimeout);

    if (computeMode == cudaComputeModeProhibited) {
        SPDLOG_ERROR("Error: device is running in <Compute Mode Prohibited>, "
                     "no threads can use cudaSetDevice().");
        assert(false);
    }

    if (major < 1) {
        SPDLOG_ERROR("gpuDeviceInit(): GPU device does not support CUDA.");
        exit(EXIT_FAILURE);
    }

    CUDA_CHK(cudaSetDevice(dev_id));
    SPDLOG_INFO("gpuDeviceInit() CUDA Device [{}]: \"{}\"", dev_id,
                _ConvertSMVer2ArchName(major, minor));

    SMPCount = prop.multiProcessorCount;

    CUDA_CHK(cudarf_cuda_malloc((void **)&d_output, width * height * sizeof(GLubyte) * 4));
    CUDA_CHK(cudaMallocHost((void **)&cpu_output, width * height * sizeof(GLubyte) * 4));

    // create texture for display
    glGenTextures(1, &gl_output.fbo_tex);
    glBindTexture(GL_TEXTURE_2D, gl_output.fbo_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

cudarf::CudaOutput::~CudaOutput()
{
    if (cpu_output) {
        CUDA_CHK(cudarf_cuda_free_host(cpu_output));
    }

    if (d_output) {
        CUDA_CHK(cudarf_cuda_free(d_output));
    }
}
