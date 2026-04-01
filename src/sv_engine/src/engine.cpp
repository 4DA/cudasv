#include <engine/engine.hpp>
#include "engine_internal.hpp"

using namespace engine;

Engine::Engine()
{
    _impl = new engine::EngineImpl();

    assert(_impl);
}

Engine::~Engine()
{
    if (_impl) {
        delete _impl;
    }
}
