#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include <engine/engine.hpp>

#include "engine_internal.hpp"

using namespace engine;

namespace
{

bool ensure_parent_directory(const std::string &path)
{
    const std::filesystem::path output_path(path);
    if (!output_path.has_parent_path()) {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_path.parent_path(), ec);
    if (ec) {
        SPDLOG_ERROR("Failed to create output directory '{}': {}",
                     output_path.parent_path().string(),
                     ec.message());
        return false;
    }

    return true;
}

} // namespace

engine::Error Engine::dump_output_png(const std::string &path, int output_index)
{
    if (path.empty()) {
        SPDLOG_ERROR("dump_output_png requires a non-empty output path");
        return BAD_PARAMETER;
    }

    if (output_index < 0 || output_index >= static_cast<int>(SV_MAX_OUTPUTS)) {
        SPDLOG_ERROR("dump_output_png output index {} is out of range", output_index);
        return BAD_PARAMETER;
    }

    cudarf::CudaOutput *cuda_output = _impl->cudaOutput[output_index].get();
    if (cuda_output == nullptr || cuda_output->d_output == nullptr) {
        SPDLOG_ERROR("dump_output_png requested inactive output {}", output_index);
        return ERROR;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(cuda_output->width) * static_cast<std::size_t>(cuda_output->height);
    const std::size_t byte_count = pixel_count * sizeof(uchar4);

    std::vector<uchar4> host_pixels(pixel_count);
    const cudaError_t cuda_status =
        cudaMemcpy(host_pixels.data(), cuda_output->d_output, byte_count, cudaMemcpyDeviceToHost);

    if (cuda_status != cudaSuccess) {
        SPDLOG_ERROR("cudaMemcpy failed in dump_output_png: {}", cudaGetErrorString(cuda_status));
        return ERROR;
    }

    if (!ensure_parent_directory(path)) {
        return ERROR;
    }

    if (stbi_write_png(path.c_str(),
                       cuda_output->width,
                       cuda_output->height,
                       4,
                       host_pixels.data(),
                       cuda_output->width * static_cast<int>(sizeof(uchar4))) == 0) {
        SPDLOG_ERROR("Failed to write PNG dump to '{}'", path);
        return ERROR;
    }

    SPDLOG_INFO("Dumped framebuffer to '{}'", path);
    return OK;
}
