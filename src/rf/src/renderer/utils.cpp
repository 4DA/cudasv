#include <cstdio>
#include <cstring>

#include <spdlog/spdlog.h>

#include <rf/renderer/glm_common.hpp>
#include "renderer/cudarf/helpers.hpp"

#include "utils.hpp"

using namespace rf;

void rf::log_duration(const std::string &message, const Duration& duration, unsigned int indent)
{
    const unsigned int PAD_MAX = 64;
    char padding[PAD_MAX];

    if (indent < PAD_MAX-1) {
        memset(padding, ' ', indent);
        padding[indent] = '\0';
    }

    SPDLOG_INFO("CPU TIME(ms) {}{}: {:.3f}", padding, message.c_str(), duration.count());
}
