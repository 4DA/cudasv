#include <algorithm>

#include <spdlog/spdlog.h>

#include <rf/renderer/trs_transform.hpp>

#include "scene.hpp"
#include "trs_transform.hpp"

using namespace rf;

Scene::Scene():
    root(SceneComponent("root"))
{
    sceneComponents[root.name] = &root;
}

void Scene::set_ibl(IBL &&iblRV) {
    ibl = std::move(iblRV);
    assert(ibl);
}

SceneComponent * Scene::get(const std::string &name) const
{
    auto pr_it = primitiveComponents.find(name);

    if (pr_it != primitiveComponents.end()) {
        return pr_it->second;
    }

    auto sc_it = sceneComponents.find(name);
    if (sc_it != sceneComponents.end()) {
        return sc_it->second;
    }

    return nullptr;
}

SceneComponent *
Scene::add_scene_component(const std::string &name, const TRSTransform &transform, SceneComponent *parent)
{
    if (parent) {
        assert(get(parent->name));
    }

    SceneComponent *new_compo = new SceneComponent(name, transform, parent);
    parent->children.push_back(new_compo);

    auto it = sceneComponents.find(name);
    assert(it == sceneComponents.end());

    sceneComponents[name] = new_compo;
    components[name] = std::unique_ptr<SceneComponent>(new_compo);
    return new_compo;
}

PrimitiveComponent *
Scene::add_primitive_component(const std::string &name,
                               const TRSTransform &transform,
                               SceneComponent *parent,
                               bool selectable,
                               bool front_facing)
{
    if (parent) {
        assert(get(parent->name));
    }

    PrimitiveComponent *new_compo =
        new PrimitiveComponent(name, transform, parent, selectable, front_facing);
    parent->children.push_back(new_compo);

    auto it = primitiveComponents.find(name);
    assert(it == primitiveComponents.end());

    primitiveComponents[name] = new_compo;
    components[name] = std::unique_ptr<SceneComponent>(new_compo);
    return new_compo;
}

PrimitiveComponent *
Scene::add_primitive_component(std::unique_ptr<PrimitiveComponent> compo,
                               SceneComponent *parent)
{
    if (parent) {
        assert(get(parent->name));
    }

    parent->children.push_back(compo.get());

    auto it = primitiveComponents.find(compo->name);
    assert(it == primitiveComponents.end());

    PrimitiveComponent *ptr = compo.get();
    primitiveComponents[compo->name] = ptr;
    components[compo->name] = std::move(compo);

    return ptr;
}

PointLightComponent *
Scene::add_light_component(const std::string &name,
                           const TRSTransform &transform,
                           float intensity,
                           SceneComponent *parent)
{
    if (parent) {
        assert(get(parent->name));
    }

    PointLightComponent *new_compo = new PointLightComponent(name, transform, parent, intensity);
    parent->children.push_back(new_compo);

    auto it = lights.find(name);
    assert(it == lights.end());

    lights[name] = new_compo;
    components[name] = std::unique_ptr<SceneComponent>(new_compo);
    return new_compo;
}

bool Scene::add_material(const std::string &name,
                         std::shared_ptr<cudarf::Material> material)
{
    auto it = materials.find(name);
    if (it == materials.end()) {
        materials[name] = material;
        return true;
    } else {
        SPDLOG_ERROR("material [name={}] already registered", name);
        return false;
    }
}

void
Scene::build_draw_list(const std::string &name,
                       render_pass_type pass,
                       cudarf::ShaderType shaderType,
                       const rf::VirtualCamera &camera,
                       const cudarf::MaterialPtrMap &ptrMap,
                       const std::set<std::string> &except,
                       cudarf::LayerComputeFn layerFn,
                       cudarf::DrawList &list) const
{
    assert(list.drawPacketIds.size() == list.matIds.size());
    assert(list.drawPacketIds.size() == list.uniforms.size());

    if (except.find(name) != except.end()) {return;}

    bool isTranslucent = (pass == RENDER_PASS_TRANSLUCENT || pass == RENDER_PASS_UI);

    auto comp_it = components.find(name);
    assert(comp_it != components.end());

    auto prim_comp_it = primitiveComponents.find(name);

    if (prim_comp_it != primitiveComponents.end()) {
        for (const auto &prim: prim_comp_it->second->get_primitives()) {
            VertexTransform VT = prim_comp_it->second->get_vertex_transform(&camera);

            cudarf::Uniforms uniforms = cudarf::make_uniforms(VT.P, VT.V, VT.M);

            // TODO implement if non-uniform scaling is applied to primitives
            // uniforms.N = VT.N;

            if (prim->cudarfMaterial->isTranslucent == isTranslucent && prim->cudarfMaterial->type == shaderType) {
                auto center = prim_comp_it->second->compute_bounding_box().center();

                list.drawPacketIds.push_back(prim->drawPacketId);
                list.matIds.push_back(ptrMap.at(prim->cudarfMaterial));
                list.uniforms.push_back(uniforms);
                list.info.push_back(
                    std::make_pair<int, cudarf::DrawList::ItemInfo>(
                        list.info.size(),
                        cudarf::DrawList::ItemInfo {
                            glm::length(camera.transform.translation - center),
                            (layerFn ? layerFn(*prim_comp_it->second) : CUDARF_LAYER_DEFAULT)
                        }
                        ));
            }
        }
    }

    for (const auto &child: comp_it->second->children) {
        build_draw_list(child->name, pass, shaderType, camera, ptrMap, except, layerFn, list);
    }
}

std::shared_ptr<cudarf::Material> Scene::get_material(const std::string &name) const
{
    auto it = materials.find(name);
    if (it == materials.end()) {
        return nullptr;
    }

    return it->second;
}

std::vector<RayIntersectionResult> Scene::ray_intersect(const glm::vec3 &O, const glm::vec3 &r, const rf::VirtualCamera *camera)
{
    std::vector<RayIntersectionResult> result;

    for (auto iter: primitiveComponents) {
        PrimitiveComponent *c = iter.second;

        if (!c->is_hitable()) continue;

        BoundingBox box = c->compute_bounding_box(camera);

        HitRecord hit;

        if (intersect(O, r, box, hit)) {
            result.push_back(RayIntersectionResult{c, hit.t});
        }
    }

    std::sort(std::begin(result), std::end(result),
              [](const RayIntersectionResult &rq1, const RayIntersectionResult &rq2)
                  { return rq1.t < rq2.t; });

    return result;
}

void cudarf::DrawList::sort(const rf::VirtualCamera &)
{
    assert(drawPacketIds.size() == uniforms.size());
    assert(drawPacketIds.size() == matIds.size());

    // this assert will flag up when sort is called on opaque list
    assert(drawPacketIds.size() == info.size());

    std::vector<unsigned int> order;
    order.reserve(drawPacketIds.size());

    std::vector<Uniforms> _uniforms(uniforms.size());
    std::vector<unsigned int> _drawPacketIds(drawPacketIds.size());
    std::vector<unsigned int> _matIds(matIds.size());

    std::sort(std::begin(info), std::end(info), [&order](const Pair &p1, const Pair &p2) -> bool {
        if (p1.second.layer < p2.second.layer) {
            return true;
        } else if (p1.second.layer == p2.second.layer) {
            return p1.second.distance > p2.second.distance;
        } else {
            return false;
        }
    });

    for (unsigned int i = 0; i < drawPacketIds.size(); i++) {
        int j = info[i].first;
        _uniforms[i] = uniforms[j];
        _drawPacketIds[i] = drawPacketIds[j];
        _matIds[i] = matIds[j];
    }

    this->uniforms = _uniforms;
    this->drawPacketIds = _drawPacketIds;
    this->matIds = _matIds;
}
