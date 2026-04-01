#ifndef CUDARF_CUBEMAP_INL
#define CUDARF_CUBEMEP_INL

__device__
float4 sampleCube(const cudarf::CubeMap &cubeMap, float x, float y, float z, float lod)
{
    return texCubemapLod<float4>(cubeMap.tex, x, y, z, lod);
}

#endif
