#include <spdlog/spdlog.h>

#ifndef RF_GLCOMMON_HPP
#define RF_GLCOMMON_HPP

#include <GLES3/gl3.h>

namespace rf
{

#define GL_CHECK(__where)                                               \
    while (true) {                                                      \
        GLenum __e = glGetError();                                      \
        if (__e != GL_NO_ERROR) {                                       \
            SPDLOG_ERROR("GL error 0x{:x} in \"{}\": ({}:{})", \
                         __e, __where, __FILE__, __LINE__);             \
            exit(0);                                                    \
        }                                                               \
        break;                                                          \
    }

#define GL_CHECK_ERROR(__where)                                         \
    do                                                                  \
    {                                                                   \
        int fail = 0;                                                   \
        while (true) {                                                  \
            GLenum __e = glGetError();                                  \
            if (__e == GL_NO_ERROR) {                                   \
                if (fail) return fail;                                  \
                break;                                                  \
            } else {                                                    \
                fail = -1;                                              \
                SPDLOG_ERROR("GL error 0x{:x} in \"{}\": ({}:{})", \
                             __e, __where, __FILE__, __LINE__);         \
           }                                                            \
        }                                                               \
    } while (0)

} // namespace rf

#endif
