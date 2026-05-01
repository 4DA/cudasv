#ifndef RF_SCENE
#define RF_SCENE

#include <ranges>
#include <memory>
#include <unordered_map>
#include <iterator>
#include <list>
#include <set>
#include <string>

#include <rf/renderer/primitive_component.hpp>
#include <rf/renderer/ibl.hpp>
#include <rf/renderer/scene_component.hpp>
#include <rf/renderer/point_light_component.hpp>
#include <rf/renderer/image.hpp>

// Rendering pass influence on rendering operations order
namespace rf
{
enum render_pass_type {
    RENDER_PASS_OPAQUE = 0,  // opaque geometry
    RENDER_PASS_TRANSLUCENT, // translucent geometry, must be rendered with
                             // alpha blending and after all opaque is done
    RENDER_PASS_COUNT
};
}

namespace cudarf
{

#define CUDARF_LAYER_LOWEST  0
#define CUDARF_LAYER_DEFAULT 2

struct DrawList {
    struct ItemInfo {
        float distance;      // distance from camera to object BB center
        unsigned int layer;  // object with lower value is drawn first
    };

    using Pair = std::pair<unsigned int, ItemInfo>;

    operator bool() const {
        return (drawPacketIds.size() > 0);
    }

    void sort(const rf::VirtualCamera &camera);

    std::vector<unsigned int> drawPacketIds;
    std::vector<unsigned int> matIds;
    std::vector<cudarf::Uniforms> uniforms;
    std::vector<Pair> info; // used for frame composition
};

    using LayerComputeFn = std::function<unsigned int(const rf::PrimitiveComponent &)>;
}

namespace rf
{
struct RayIntersectionResult
{
    // closest object found. null if search was unsuccessful
    PrimitiveComponent *object;

    // distance to object
    float t;
};

    // Scene stores components and resources needed to render them
    //
class Scene {
    public:

    Scene();

    // Retrieve all objects that the ray starting at 'from' with direction 'dir' intersects.
    // - from: Origin from which the ray is shot
    // - dir: Normalized direction vector
    // - returns: RayIntersectionResult containing the search results
    std::vector<RayIntersectionResult> ray_intersect(const glm::vec3 &from,
                                                     const glm::vec3 &dir,
                                                     const rf::VirtualCamera *camera);

    // Retrieve a scene component by its name.
    // - name: The name of the component
    SceneComponent * get(const std::string &name) const;

    // Retrieve the root component.
    SceneComponent * get_root() {return &root;}

    // Retrieve a material by its name.
    // - name: The name of the material
    std::shared_ptr<cudarf::Material> get_material(const std::string &name) const;

    unsigned int get_materials_count() const {return materials.size();}

    auto get_materials() const {return std::ranges::subrange{materials};}

    auto get_lights() const {return std::ranges::subrange{lights};}

    const rf::IBL & get_ibl() const {return ibl;}

    // Create a new scene component and add it to the scene.
    // - name: The name of the new component. The scene shouldn't already contain
    // a component with this name.
    // - transform: The initial transform of the component.
    SceneComponent *
    add_scene_component(const std::string &name, const TRSTransform &transform, SceneComponent *parent);

    // Create a new primitive component and add it to the scene.
    // - name: The name of the new component. The scene shouldn't already contain
    //         a component with this name.
    // - transform: The initial transform of the component.
    // - hitable: If true, the primitive accepts camera rays.
    // - frontFacing: If true, the primitive should be oriented towards
    //                 the camera with its local +X axis.
    // - parent: Pointer to the parent component.
    PrimitiveComponent *
    add_primitive_component(const std::string &name,
                            const TRSTransform &transform,
                            SceneComponent *parent,
                            bool hitable = false,
                            bool frontFacing = false);

    PrimitiveComponent *
    add_primitive_component(std::unique_ptr<PrimitiveComponent> compo, SceneComponent *parent);

    // Create a new light component and add it to the scene.
    // - name: The name of the new component. The scene shouldn't already contain
    // a component with this name.
    // - transform: The initial transform of the component.
    // - intensity: Radiant intensity (in watts).
    // - parent: Pointer to the parent component.
    PointLightComponent *
    add_light_component(const std::string &name, const TRSTransform &transform, float intensity, SceneComponent *parent);

    bool add_material(const std::string &name,
                      std::shared_ptr<cudarf::Material> material);

    SceneComponent * get_scene_component(const std::string &name) const {
        auto it = sceneComponents.find(name);
        if (it == sceneComponents.end()) {return nullptr;}
        else {return it->second;}
    }

    void set_ibl(IBL && iblRV);

    void build_draw_list(const std::string &name,
                         render_pass_type type,
                         cudarf::ShaderType shaderType,
                         const VirtualCamera &camera,
                         const cudarf::MaterialPtrMap &ptrMap,
                         const std::set<std::string> &except,
                         cudarf::LayerComputeFn layerFn,
                         cudarf::DrawList &output) const;

private:
    SceneComponent root;

    cudarf::MaterialNames materials;

    std::unordered_map<std::string, std::unique_ptr<SceneComponent>> components;
    std::unordered_map<std::string, SceneComponent *> sceneComponents;
    std::unordered_map<std::string, PrimitiveComponent *> primitiveComponents;
    std::unordered_map<std::string, PointLightComponent *> lights;

    IBL ibl;
};

} // namespace rf

#endif
