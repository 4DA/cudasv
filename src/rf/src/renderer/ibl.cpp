#include <cctype>
#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

#include "renderer/cuda_helpers.hpp"
#include "texture.hpp"

#include "ibl.hpp"

const unsigned int CUBE_FACES = 6;

using namespace rf;

SphericalHarmonics SphericalHarmonics::load_from_file(const std::string &file)
{
    std::vector<glm::vec3> result;
    std::ifstream fin(file, std::ios::in);
    std::size_t shLine = 0;
    std::size_t fileLine = 0;

    if (!fin.good()) {
        SPDLOG_ERROR("Error opening {}", file.c_str());
        return SphericalHarmonics();
    }

    std::string line;
    while (shLine < SphericalHarmonics::ROWS_COUNT && std::getline(fin, line)) {
        fileLine++;

        const std::size_t commentPos = line.find("//");
        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }

        for (char &ch : line) {
            if (!(std::isdigit(static_cast<unsigned char>(ch)) ||
                  ch == '+' || ch == '-' || ch == '.' || ch == 'e' || ch == 'E')) {
                ch = ' ';
            }
        }

        std::istringstream iss(line);
        float R = 0.0f;
        float G = 0.0f;
        float B = 0.0f;
        if (!(iss >> R >> G >> B)) {
            continue;
        }

        SPDLOG_INFO("SH coefs({}): {:f}, {:f}, {:f}", shLine, R, G, B);
        result.push_back(glm::vec3(R, G, B));
        shLine++;
    }

   if (result.size() != SphericalHarmonics::ROWS_COUNT) {
        SPDLOG_ERROR("Incorrect SH rows count: {} (expected {}) in {}", result.size(), SphericalHarmonics::ROWS_COUNT, file.c_str());
    }

    return SphericalHarmonics{result};
}

static cudarf::TextureResource create_cubemap_mipped(
    std::vector<CubemapDescription> specularMap,
    cudaStream_t cuStream)
{
    int lod = 0;
    const auto mipLevels = static_cast<unsigned int>(specularMap.size());
    CubemapDescription topLod = specularMap[lod];
    int width = topLod[0].w;
    assert(topLod[0].channels == 4);

    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    cudaMipmappedArray *dev_mipmapArray;
    CUDA_CHK(cudaMallocMipmappedArray(&dev_mipmapArray,
                                      &channelDesc,
                                      make_cudaExtent(width, width, CUBE_FACES),
                                      mipLevels,
                                      cudaArrayCubemap));

    for (unsigned int id = 0; id < mipLevels; id++) {
        CubemapDescription lod = specularMap[id];
        int width = lod[0].w;

        auto data = std::make_unique<float[]>(4 * width * width * CUBE_FACES);

        for (unsigned int i = 0; i < CUBE_FACES; i++) {
            memcpy(&data[4 * width * width * i], lod[i].data, 4 * sizeof(float) * width * width);
        }

        cudaArray_t dev_mipLevelArray;

        CUDA_CHK(cudaGetMipmappedArrayLevel(&dev_mipLevelArray, dev_mipmapArray, id));

        // DEBUG: to get mip level properties
        // cudaChannelFormatDesc desc;
        // cudaExtent extent;
        // unsigned int flags;
        // CUDA_CHK(cudaArrayGetInfo(&desc, &extent, &flags, dev_mipLevelArray));

        cudaMemcpy3DParms myparms = {0};
        myparms.srcPos   = make_cudaPos(0, 0, 0);
        myparms.dstPos   = make_cudaPos(0, 0, 0);
        myparms.srcPtr   = make_cudaPitchedPtr(data.get(), width * sizeof(float) * 4, width, width);
        myparms.dstArray = dev_mipLevelArray;
        myparms.extent   = make_cudaExtent(width, width, CUBE_FACES);
        myparms.kind     = cudaMemcpyHostToDevice;
        CUDA_CHK(cudaMemcpy3DAsync(&myparms, cuStream));
    }

    cudaTextureObject_t tex;
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeMipmappedArray;
    texRes.res.mipmap.mipmap = dev_mipmapArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords    = true;
    texDescr.filterMode          = cudaFilterModeLinear;
    texDescr.mipmapFilterMode    = cudaFilterModeLinear;
    texDescr.addressMode[0]      = cudaAddressModeMirror;
    texDescr.addressMode[1]      = cudaAddressModeMirror;
    texDescr.addressMode[2]      = cudaAddressModeClamp;
    texDescr.readMode            = cudaReadModeElementType;
    texDescr.maxMipmapLevelClamp = float(mipLevels - 1);

    // for TextureDesc_v2:
    // texDescr.seamlessCubeMap = 1;

    CUDA_CHK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    auto view = cudarf::Texture{
        tex,
        topLod[0].channels,
        topLod[0].w,
        topLod[0].h,
        mipLevels,
        false,
        glm::mat3(1.0f)
    };

    return cudarf::TextureResource(view,
                                   nullptr,
                                   dev_mipmapArray);
}

rf::IBL rf::load_ibl(const std::string &path_prefix, cudaStream_t cuStream) {
    SphericalHarmonics sphericalHarmonics;

    if (std::ifstream(path_prefix + "ibl/diffuse/sh.txt").good()) {
        sphericalHarmonics = SphericalHarmonics::load_from_file(path_prefix + "ibl/diffuse/sh.txt");
    }

    SPDLOG_INFO("Using spherical harmonics");

    std::vector<CubemapDescription> specularMap;

    // todo: levels should be determined automatically
    for (unsigned int i = 0; i < 9; i++) {
        CubemapDescription specularLevel = {
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_px.hdr")),
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_nx.hdr")),
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_py.hdr")),
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_ny.hdr")),
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_pz.hdr")),
            load_image_hdr(std::string(path_prefix + "ibl/specular/m") + std::to_string(i) + std::string("_nz.hdr"))
        };
        specularMap.push_back(specularLevel);
    }

    auto specularTex = create_cubemap_mipped(specularMap, cuStream);

    for (unsigned int i = 0; i < specularMap.size(); i++) {
        CubemapDescription specularLevel = specularMap[i];
        for (size_t j = 0; j < specularLevel.size(); j++) {
            free((void*)specularLevel[j].data);
        }
    }

    Image lutDesc = load_image(path_prefix + "ibl/brdfLUT_16.png", true, false, 4);

    auto lutTex = cudarf::create_cuda_texture(lutDesc, cudaAddressModeClamp, 1, std::nullopt, cuStream);
    assert(lutTex);

    free((void*)lutDesc.data);

    return IBL(sphericalHarmonics, std::move(*lutTex), std::move(specularTex));
}
