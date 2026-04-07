#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>

#include <GLES3/gl3.h>
#include <spdlog/spdlog.h>

#include "app/nuscenes_inspector.hpp"
#include "sources/nuscenes_source.hpp"
#include "sources/source_validation.hpp"

namespace
{

struct InspectorControlState
{
    bool leftWasPressed = false;
    bool rightWasPressed = false;
    bool homeWasPressed = false;
    bool endWasPressed = false;
    bool leftShiftWasPressed = false;
    bool rightShiftWasPressed = false;
    bool pWasPressed = false;
    bool spaceWasPressed = false;
    std::array<bool, 6> digitWasPressed = {};
};

struct MosaicSlot
{
    camera::CameraRole role = camera::CameraRole::Unknown;
    GLuint texture = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

static const char *camera_role_to_string(camera::CameraRole role)
{
    switch (role) {
    case camera::CameraRole::Front:
        return "front";
    case camera::CameraRole::Rear:
        return "rear";
    case camera::CameraRole::FrontLeft:
        return "front_left";
    case camera::CameraRole::FrontRight:
        return "front_right";
    case camera::CameraRole::RearLeft:
        return "rear_left";
    case camera::CameraRole::RearRight:
        return "rear_right";
    case camera::CameraRole::Left:
    case camera::CameraRole::Right:
    case camera::CameraRole::Unknown:
        break;
    }

    return "unknown";
}

static const char *vertex_shader_source()
{
    return R"(#version 300 es
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;
out vec2 vTexCoord;
void main()
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    vTexCoord = inTexCoord;
}
)";
}

static const char *fragment_shader_source()
{
    return R"(#version 300 es
precision mediump float;
in vec2 vTexCoord;
uniform sampler2D uTexture;
out vec4 outColor;
void main()
{
    // Decoded image buffers are laid out top-to-bottom in memory, while
    // the simple fullscreen quad uses bottom-to-top texture coordinates.
    outColor = texture(uTexture, vec2(vTexCoord.x, 1.0 - vTexCoord.y));
}
)";
}

static bool compile_shader(GLenum type,
                           const char *source,
                           GLuint &shader,
                           std::string &errorMessage)
{
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compileStatus = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus == GL_TRUE) {
        return true;
    }

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::string infoLog(static_cast<std::size_t>(logLength), '\0');
    glGetShaderInfoLog(shader, logLength, nullptr, infoLog.data());
    errorMessage = infoLog;
    glDeleteShader(shader);
    shader = 0;
    return false;
}

static bool build_program(GLuint &program, std::string &errorMessage)
{
    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
    if (!compile_shader(GL_VERTEX_SHADER, vertex_shader_source(), vertexShader, errorMessage)) {
        return false;
    }
    if (!compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source(), fragmentShader, errorMessage)) {
        glDeleteShader(vertexShader);
        return false;
    }

    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus == GL_TRUE) {
        return true;
    }

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::string infoLog(static_cast<std::size_t>(logLength), '\0');
    glGetProgramInfoLog(program, logLength, nullptr, infoLog.data());
    errorMessage = infoLog;
    glDeleteProgram(program);
    program = 0;
    return false;
}

static int find_camera_index(const videoio::FramePacket &packet, camera::CameraRole role)
{
    for (std::size_t index = 0; index < packet.cameras.size(); ++index) {
        if (packet.cameras[index].role == role) {
            return static_cast<int>(index);
        }
    }

    return -1;
}

static std::string build_window_title(const svapp::NuScenesSource &source,
                                      const videoio::FramePacket &packet,
                                      bool autoplayEnabled,
                                      bool focusModeEnabled,
                                      camera::CameraRole focusedRole)
{
    std::ostringstream title;
    title << "NuScenes Inspector "
          << "[" << (source.current_sample_index() + 1)
          << "/" << source.sample_count() << "] ";

    if (packet.metadata.has_sample_id) {
        title << packet.metadata.sample_id;
    } else {
        title << "<no-sample-id>";
    }

    title << " | ";
    title << (autoplayEnabled ? "play" : "pause");
    title << " | ";
    if (focusModeEnabled) {
        title << "focus:" << camera_role_to_string(focusedRole);
    } else {
        title << "mosaic";
    }

    return title.str();
}

static void set_aspect_fit_viewport(int originX,
                                    int originY,
                                    int targetWidth,
                                    int targetHeight,
                                    uint32_t sourceWidth,
                                    uint32_t sourceHeight)
{
    if (targetWidth <= 0 || targetHeight <= 0 ||
        sourceWidth == 0 || sourceHeight == 0) {
        glViewport(originX, originY, targetWidth, targetHeight);
        return;
    }

    const float targetAspect = static_cast<float>(targetWidth) / static_cast<float>(targetHeight);
    const float sourceAspect = static_cast<float>(sourceWidth) / static_cast<float>(sourceHeight);

    int viewportWidth = targetWidth;
    int viewportHeight = targetHeight;
    int viewportX = originX;
    int viewportY = originY;

    // Letterbox each camera tile instead of stretching the decoded image,
    // so camera geometry stays visually faithful during inspection.
    if (sourceAspect > targetAspect) {
        viewportHeight = static_cast<int>(static_cast<float>(targetWidth) / sourceAspect);
        viewportY += (targetHeight - viewportHeight) / 2;
    } else {
        viewportWidth = static_cast<int>(static_cast<float>(targetHeight) * sourceAspect);
        viewportX += (targetWidth - viewportWidth) / 2;
    }

    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
}

} // namespace

namespace svapp
{

int run_nuscenes_inspector_loop(AppContext &app,
                                GLFWHost &glfwHost,
                                const CmdlineOpts &options,
                                const engine::OutputSet &outputSet,
                                videoio::FrameSource &frameSource)
{
    (void)options;

    app.interactive_input_enabled = false;
    auto *nuScenesSource = dynamic_cast<NuScenesSource *>(&frameSource);
    if (!nuScenesSource) {
        SPDLOG_ERROR("NuScenes inspector requires a NuScenesSource instance");
        return EXIT_FAILURE;
    }

    constexpr std::array<camera::CameraRole, 6> kMosaicRoles = {
        camera::CameraRole::FrontLeft,
        camera::CameraRole::Front,
        camera::CameraRole::FrontRight,
        camera::CameraRole::RearLeft,
        camera::CameraRole::Rear,
        camera::CameraRole::RearRight,
    };

    int outputIndex = 0;
    for (unsigned int index = 0; index < SV_MAX_OUTPUTS; ++index) {
        if (outputSet.outputs[index].active) {
            outputIndex = static_cast<int>(index);
            break;
        }
    }

    glfwHost.make_current(outputIndex);

    std::string errorMessage;
    GLuint program = 0;
    if (!build_program(program, errorMessage)) {
        SPDLOG_ERROR("Failed to build NuScenes inspector shader program: {}", errorMessage);
        return EXIT_FAILURE;
    }

    static constexpr float kQuadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
    };

    GLuint vertexArray = 0;
    GLuint vertexBuffer = 0;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glBindVertexArray(vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    std::array<MosaicSlot, 6> slots = {};
    for (std::size_t index = 0; index < slots.size(); ++index) {
        slots[index].role = kMosaicRoles[index];
        glGenTextures(1, &slots[index].texture);
        glBindTexture(GL_TEXTURE_2D, slots[index].texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        const std::array<uint8_t, 3> blackPixel = {0, 0, 0};
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGB,
                     1,
                     1,
                     0,
                     GL_RGB,
                     GL_UNSIGNED_BYTE,
                     blackPixel.data());
    }

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTexture"), 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    uint64_t uploadedSourceSequence = std::numeric_limits<uint64_t>::max();
    bool firstPacketReported = false;
    InspectorControlState controls = {};
    bool autoplayEnabled = false;
    bool focusModeEnabled = false;
    std::size_t focusedSlotIndex = 1;
    constexpr int kFastStepCount = 5;
    constexpr double kAutoplayStepIntervalSec = 0.25;
    constexpr useconds_t kInspectorLoopSleepUs = 16000;
    double lastAutoplayStepTime = glfwGetTime();

    SPDLOG_INFO("NuScenes inspector controls: Left/Right arrows step through scene samples, "
                "Shift+Left/Right jump by {}, Home/End jump to scene bounds, "
                "P toggles autoplay, 1-6 select a camera, Space toggles focused view",
                kFastStepCount);

    while (app.running && !glfwHost.should_close_any()) {
        const bool leftPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_LEFT);
        const bool rightPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_RIGHT);
        const bool homePressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_HOME);
        const bool endPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_END);
        const bool leftShiftPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_LEFT_SHIFT);
        const bool rightShiftPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_RIGHT_SHIFT);
        const bool pPressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_P);
        const bool spacePressed = glfwHost.key_pressed(outputIndex, GLFW_KEY_SPACE);

        const bool fastStepRequested = leftShiftPressed || rightShiftPressed;
        if (leftPressed && !controls.leftWasPressed) {
            nuScenesSource->step_samples(fastStepRequested ? -kFastStepCount : -1);
        }
        if (rightPressed && !controls.rightWasPressed) {
            nuScenesSource->step_samples(fastStepRequested ? kFastStepCount : 1);
        }
        if (homePressed && !controls.homeWasPressed) {
            nuScenesSource->set_sample_index(0);
        }
        if (endPressed && !controls.endWasPressed && nuScenesSource->sample_count() > 0) {
            nuScenesSource->set_sample_index(nuScenesSource->sample_count() - 1);
        }
        if (pPressed && !controls.pWasPressed) {
            autoplayEnabled = !autoplayEnabled;
            lastAutoplayStepTime = glfwGetTime();
            SPDLOG_INFO("NuScenes inspector autoplay: {}",
                        autoplayEnabled ? "enabled" : "disabled");
        }
        if (spacePressed && !controls.spaceWasPressed) {
            focusModeEnabled = !focusModeEnabled;
            SPDLOG_INFO("NuScenes inspector view mode: {}",
                        focusModeEnabled
                            ? camera_role_to_string(slots[focusedSlotIndex].role)
                            : "mosaic");
        }

        controls.leftWasPressed = leftPressed;
        controls.rightWasPressed = rightPressed;
        controls.homeWasPressed = homePressed;
        controls.endWasPressed = endPressed;
        controls.leftShiftWasPressed = leftShiftPressed;
        controls.rightShiftWasPressed = rightShiftPressed;
        controls.pWasPressed = pPressed;
        controls.spaceWasPressed = spacePressed;

        for (std::size_t slotIndex = 0; slotIndex < slots.size(); ++slotIndex) {
            const int glfwKey = GLFW_KEY_1 + static_cast<int>(slotIndex);
            const bool digitPressed = glfwHost.key_pressed(outputIndex, glfwKey);
            if (digitPressed && !controls.digitWasPressed[slotIndex]) {
                focusedSlotIndex = slotIndex;
                SPDLOG_INFO("NuScenes inspector selected camera [{}]: {}",
                            slotIndex + 1,
                            camera_role_to_string(slots[slotIndex].role));
            }
            controls.digitWasPressed[slotIndex] = digitPressed;
        }

        const double currentTime = glfwGetTime();
        if (autoplayEnabled && (currentTime - lastAutoplayStepTime) >= kAutoplayStepIntervalSec) {
            nuScenesSource->step_next_sample();
            lastAutoplayStepTime = currentTime;
        }

        videoio::FramePacket packet;
        if (!frameSource.get_next_frame(packet)) {
            SPDLOG_ERROR("Failed to fetch NuScenes source packet");
            return EXIT_FAILURE;
        }

        if (!firstPacketReported) {
            report_source_packet(packet);
            SPDLOG_INFO("NuScenes inspector sample [{} / {}]: {}",
                        nuScenesSource->current_sample_index() + 1,
                        nuScenesSource->sample_count(),
                        packet.metadata.sample_id);
            firstPacketReported = true;
        }

        glfwHost.set_window_title(outputIndex,
                                  build_window_title(*nuScenesSource,
                                                     packet,
                                                     autoplayEnabled,
                                                     focusModeEnabled,
                                                     slots[focusedSlotIndex].role));

        if (uploadedSourceSequence != packet.metadata.source_frame_sequence) {
            for (auto &slot : slots) {
                const int cameraIndex = find_camera_index(packet, slot.role);
                if (cameraIndex < 0) {
                    SPDLOG_ERROR("NuScenes source packet is missing role '{}'",
                                 camera_role_to_string(slot.role));
                    frameSource.release_frame(packet);
                    return EXIT_FAILURE;
                }

                const auto &cameraFrame = packet.cameras[static_cast<std::size_t>(cameraIndex)];
                glBindTexture(GL_TEXTURE_2D, slot.texture);
                // Upload the decoded camera image into the inspector texture.
                if (slot.width != cameraFrame.width || slot.height != cameraFrame.height) {
                    slot.width = cameraFrame.width;
                    slot.height = cameraFrame.height;
                    glTexImage2D(GL_TEXTURE_2D,
                                 0,
                                 GL_RGB,
                                 static_cast<GLsizei>(cameraFrame.width),
                                 static_cast<GLsizei>(cameraFrame.height),
                                 0,
                                 GL_RGB,
                                 GL_UNSIGNED_BYTE,
                                 cameraFrame.data);
                } else {
                    glTexSubImage2D(GL_TEXTURE_2D,
                                    0,
                                    0,
                                    0,
                                    static_cast<GLsizei>(cameraFrame.width),
                                    static_cast<GLsizei>(cameraFrame.height),
                                    GL_RGB,
                                    GL_UNSIGNED_BYTE,
                                    cameraFrame.data);
                }
            }

            uploadedSourceSequence = packet.metadata.source_frame_sequence;
        }

        glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(vertexArray);

        const int displayWidth = outputSet.outputs[outputIndex].config.display_width;
        const int displayHeight = outputSet.outputs[outputIndex].config.display_height;
        const int tileWidth = displayWidth / 3;
        const int tileHeight = displayHeight / 2;

        if (focusModeEnabled) {
            set_aspect_fit_viewport(0,
                                    0,
                                    displayWidth,
                                    displayHeight,
                                    slots[focusedSlotIndex].width,
                                    slots[focusedSlotIndex].height);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, slots[focusedSlotIndex].texture);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        } else {
            for (std::size_t index = 0; index < slots.size(); ++index) {
                // Arrange the six cameras into a 3x2 grid:
                // front row on top, rear row on bottom.
                const int column = static_cast<int>(index % 3);
                const int row = 1 - static_cast<int>(index / 3);
                set_aspect_fit_viewport(column * tileWidth,
                                        row * tileHeight,
                                        tileWidth,
                                        tileHeight,
                                        slots[index].width,
                                        slots[index].height);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, slots[index].texture);
                glDrawArrays(GL_TRIANGLES, 0, 6);
            }
        }

        glfwHost.swap_buffers(outputIndex);

        if (!frameSource.release_frame(packet)) {
            SPDLOG_ERROR("Failed to release NuScenes source packet");
            return EXIT_FAILURE;
        }

        usleep(kInspectorLoopSleepUs);
    }

    return 0;
}

} // namespace svapp
