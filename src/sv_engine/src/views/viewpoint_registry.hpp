#ifndef SV_VIEWPOINT_REGISTRY_HPP
#define SV_VIEWPOINT_REGISTRY_HPP

#include <optional>
#include <vector>

#include <engine/views_config.hpp>

#include <rf/camera_control/viewpoint_animation.hpp>

namespace engine::view
{

class ViewpointRegistry
{
public:
    int init(const ViewConfig3D &view_config, unsigned int width, unsigned int height);

    const std::vector<rf::Viewpoint> &viewpoints() const { return _viewpoints; }

private:
    static std::optional<rf::Viewpoint> create_viewpoint(const ViewConfig3D &view_config,
                                                         const Viewpoint3D &preset,
                                                         unsigned int width,
                                                         unsigned int height);

    std::vector<rf::Viewpoint> _viewpoints;
};

} // namespace engine::view

#endif
