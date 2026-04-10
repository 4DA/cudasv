// -*- mode: c++ -*-

/* This file uses code from cudarast renderer hosted on google code:
   https://code.google.com/archive/p/cudaraster/
 
   "High-Performance Software Rasterization on GPUs",
   Samuli Laine and Tero Karras,
   Proc. High-Performance Graphics 2011
   http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
 
   Original code is licensed under New BSD License
 */   

/*******************************************************************************
 * Copyright (c) 2009-2011, NVIDIA Corporation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of NVIDIA Corporation nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

#ifndef CUDARF_TRIANGLE_INL
#define CUDARF_TRIANGLE_INL

/// find interval in prefix sum array which what element `i' belongs to
///
__inline__ __device__ int find_interval(unsigned int i, const unsigned int *offsetArray, int N)
{
    int vL = 0;
    int vR = N - 1;

    while(true) {
        int p = (vL + vR) / 2;

        if (vL == vR) {
            return vL;
            break;
        }
        else if (vL == vR - 1) {
            if (i < offsetArray[vR]) {
                vR = vL;
            }
            else {
                vL = vR;
            }
        }
        else if (i < offsetArray[p]) {
            vR = p - 1;
        } else if (i > offsetArray[p]) {
            vL = p;
        } else {
            vL = p;
            vR = p;
        }
        if (vL > vR) {
            return -1;
        }
    }
}

// DEBUG: simple linear search
//
// for (int v = 0; v < pipe->drawPacketCount; v++) {
//     int drawPacketId = pipe->drawPacketOrder[v];
//     if (i >= pipe->offsets[v] &&
//         i < pipe->offsets[v] + pipe->drawPackets[drawPacketId].index_count)
//     {
//         idx = (i - pipe->offsets[v]);
//         packetId = drawPacketId;
//         inOft = pipe->offsets[v];
//         materialId = pipe->drawPacketMaterials[v];
//         break;
//     }
// }

// CUDARF_STAGE: vertex transform

template<cudarf::ShaderType TShaderType, bool TWithTex, bool TWithTAA>
__global__
void vertex_transform(const cudarf::rast::PipeParams *pipe,
                      unsigned int vertex_count)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= vertex_count) {return;}

    // determine what draw packet and vertex index this thread should pick
    int v = find_interval(i,  pipe->vtxOffsets, pipe->drawPacketCount);
    int vIdx = i - pipe->vtxOffsets[v];
    int drawPacketId = pipe->drawPacketOrder[v];
    assert(v < pipe->drawPacketCount);
    assert(drawPacketId >= 0);

    const cudarf::Uniforms &uniforms = pipe->uniforms[v];

    if (i >= vertex_count) { return; }

    cudarf::rast::VertexIn *vertexBuffer = pipe->drawPackets[drawPacketId].dev_bufVertex;
    // TODO normals must be transformed by inverse transposed matrix
    cudarf::Vec4f N = uniforms.M * make_vec4f(vertexBuffer[vIdx].nor, 0.0f);

    glm::mat4 PVM;

#ifdef WITH_TAA
    if (TWithTAA) {
        const cudarf::Uniforms &uniformsHist = pipe->taa.uniformsHist[v];
        PVM = pipe->common.PV * uniforms.M;
        glm::mat4 PVM_hist = pipe->taa.commonHist.PV * uniformsHist.M;

        // compute screen space position of vertex in history frame
        float4 pos_3dhp_hist = PVM_hist * make_vec4f(vertexBuffer[vIdx].pos, 1.0f);
        pipe->vertexOut[i].pos_ss_hist = clip_to_window(to_vec2f(pos_3dhp_hist / pos_3dhp_hist.w),
                                                         pipe->windowWidth, pipe->windowHeight);
    } else {
        PVM = uniforms.PVM;
    }
#else
    PVM = uniforms.PVM;
#endif

    pipe->vertexOut[i].pos_3dhp = PVM * make_vec4f(vertexBuffer[vIdx].pos, 1.0f);

    if (TShaderType == cudarf::SHADER_TYPE_PBR) {
        pipe->vertexOut[i].pos_world = to_vec3f(uniforms.M * make_vec4f(vertexBuffer[vIdx].pos, 1.0f));
        pipe->vertexOut[i].nor = normalize(to_vec3f(N));
    }
    else if (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
        pipe->vertexOut[i].col = vertexBuffer[vIdx].col;
    }

    if (TWithTex) {
        pipe->vertexOut[i].tex = to_float2(vertexBuffer[vIdx].tex);
    }

}

// TODO: write possible ranges for vX.w value
// todo: justify why (CUDARF_SUBPIXEL_LOG2 - 1), eg because we want 0,0 to be
// rasterizer coords center
__device__ __inline__ void tri_to_fixed(
    const cudarf::rast::PipeParams *pipe,
    float4 v0, float4 v1, float4 v2,
    int2 &p0, int2 &p1, int2 &p2, float3 &w_rcp, int2 &lo, int2 &hi)
{
    float view_scale_x = (float)(pipe->windowWidth  << (CUDARF_SUBPIXEL_LOG2 - 1));
    float view_scale_y = (float)(pipe->windowHeight << (CUDARF_SUBPIXEL_LOG2 - 1));
    w_rcp = make_float3(1.0f / v0.w, 1.0f / v1.w, 1.0f / v2.w);
    p0 = make_int2(ptx_f32_to_s32_sat(v0.x * w_rcp.x * view_scale_x),
                   ptx_f32_to_s32_sat(v0.y * w_rcp.x * view_scale_y));
    p1 = make_int2(ptx_f32_to_s32_sat(v1.x * w_rcp.y * view_scale_x),
                   ptx_f32_to_s32_sat(v1.y * w_rcp.y * view_scale_y));
    p2 = make_int2(ptx_f32_to_s32_sat(v2.x * w_rcp.z * view_scale_x),
                   ptx_f32_to_s32_sat(v2.y * w_rcp.z * view_scale_y));
    lo = make_int2(ptx_min_min(p0.x, p1.x, p2.x), ptx_min_min(p0.y, p1.y, p2.y));
    hi = make_int2(ptx_max_max(p0.x, p1.x, p2.x), ptx_max_max(p0.y, p1.y, p2.y));
}

// Keep this long set of args instead of VertexOut ptr, because when we
// implement clipping we will need it anyway

template<cudarf::ShaderType TShaderType, bool TWithTex>
__device__ __inline__
void setup_triangle(const cudarf::rast::PipeParams *pipe,
                    cudarf::rast::Triangle *out,
                    float3 v_world0,
                    float3 v_world1,
                    float3 v_world2,
                    float4 v0, float4 v1, float4 v2,
#ifdef WITH_TAA
                    float2 v0_ss_hist,
                    float2 v1_ss_hist,
                    float2 v2_ss_hist,
#endif
                    float3 n0, float3 n1, float3 n2,
                    float2 t0, float2 t1, float2 t2,
                    cudarf::Color c0, cudarf::Color c1, cudarf::Color c2,
                    cudarf::Vec2f sP0, cudarf::Vec2f sP1, cudarf::Vec2f sP2,
                    float3 w_rcp, float area_rcp, unsigned int materialId)
{

    float2 screen = make_float2((float)pipe->windowWidth, (float)pipe->windowHeight);

    out->sP0 = sP0;
    out->sP1 = sP1;
    out->sP2 = sP2;

    if (TShaderType == cudarf::SHADER_TYPE_PBR) {
        out->normal[0] = n0;
        out->normal[1] = n1;
        out->normal[2] = n2;

        out->v_world[0] = v_world0;
        out->v_world[1] = v_world1;
        out->v_world[2] = v_world2;
    }

    if (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
        out->col[0] = c0;
        out->col[1] = c1;
        out->col[2] = c2;
    }

#ifdef WITH_TAA
    out->v_ss_hist[0] = v0_ss_hist;
    out->v_ss_hist[1] = v1_ss_hist;
    out->v_ss_hist[2] = v2_ss_hist;
#endif

    if (TWithTex) {
        out->tex[0] = t0;
        out->tex[1] = t1;
        out->tex[2] = t2;
    }

    out->area_rcp = area_rcp;

    out->w_rcp = w_rcp;

    out->zw = make_vec3f(v0.z * w_rcp.x, v1.z * w_rcp.y, v2.z * w_rcp.z);

    out->zw_min = min(min(out->zw.x, out->zw.y), out->zw.z);

    // out->izw.x = Z_to_fixed(v0.z * w_rcp.x);
    // out->izw.y = Z_to_fixed(v1.z * w_rcp.y);
    // out->izw.z = Z_to_fixed(v2.z * w_rcp.z);

    // out->debug = make_vec4f(P0.x, P0.y, 0.0f, 0.0f);

    // out->vidx = vidx;

    compute_aabb_screen(pipe, *out, out->flo, out->fhi);

    out->materialId = materialId;

    // TODO this optimization produces artifacats:
    // some fragments on tri borders are not rasterized
    // compute_bary_affine_tr(P0, P1, P2, area_rcp,
    //                        out->bary_scale_x,
    //                        out->bary_scale_y,
    //                        out->bary_bias);

    // out->aabb = get_aabb(cudarf::Vec3f(P0, 1.0f), cudarf::Vec3f(P1, 1.0f), cudarf::Vec3f(P2, 1.0f));
}

// CUDARF_STAGE: triangle setup

template<cudarf::ShaderType TShaderType, bool TWithTex>
__global__
void triangle_assembly(const cudarf::rast::PipeParams *pipe,
                       unsigned int triangle_count)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= triangle_count) {return;}

    int v = find_interval(3 * i,  pipe->idxOffsets, pipe->drawPacketCount);
    int drawPacketId = pipe->drawPacketOrder[v];
    assert(drawPacketId >= 0);
    assert(drawPacketId < CUDARF_MAX_DRAW_PACKETS);

    unsigned int idx = 3 * i - pipe->idxOffsets[v];
    unsigned int vtxOft = pipe->vtxOffsets[v];

    cudarf::rast::VertexOut *vtx = &pipe->vertexOut[vtxOft];
    cudarf::PrimitiveIndex *index = pipe->drawPackets[drawPacketId].dev_bufIdx;

    {
        int3 vidx;
        if (index == nullptr) {
            vidx = make_int3(idx, idx + 1, idx + 2);
        }
        else {
            vidx = make_int3(index[idx], index[idx + 1], index[idx + 2]);
        }

        assert(vidx.x < pipe->drawPackets[drawPacketId].vertCount);
        assert(vidx.y < pipe->drawPackets[drawPacketId].vertCount);
        assert(vidx.z < pipe->drawPackets[drawPacketId].vertCount);

        float4 v0 = vtx[vidx.x].pos_3dhp;
        float4 v1 = vtx[vidx.y].pos_3dhp;
        float4 v2 = vtx[vidx.z].pos_3dhp;

        float3 w_rcp;

        // CUDARF_STAGE: culling

        // TODO cull triangles if AABB falls between samples

        // cull triangle if all vertices are outside view frustum
        // -1 <= x/w <= 1 => -w <= x <= w
        // -1 <= y/w <= 1 => -w <= y <= w
        // -1 <= z/w <= 1 => -w <= z <= w

        if (v0.w < std::abs(v0.x) | v0.w < std::abs(v0.y) | v0.w < std::abs(v0.z))
        {
            if ((v0.w < +v0.x & v1.w < +v1.x & v2.w < +v2.x) |
                (v0.w < -v0.x & v1.w < -v1.x & v2.w < -v2.x) |
                (v0.w < +v0.y & v1.w < +v1.y & v2.w < +v2.y) |
                (v0.w < -v0.y & v1.w < -v1.y & v2.w < -v2.y) |
                (v0.w < +v0.z & v1.w < +v1.z & v2.w < +v2.z) |
                (v0.w < -v0.z & v1.w < -v1.z & v2.w < -v2.z))
            {
                return;
            }
        }

        // check if triangle is inside depth range
        if (v0.w >= std::abs(v0.z) & v1.w >= std::abs(v1.z) & v2.w >= std::abs(v2.z))
        {
            // Inside S16 range and small enough => fast path.
            // unused for now
            int2 p0, p1, p2, lo, hi;

            tri_to_fixed(pipe,
                         v0, v1, v2,
                         p0, p1, p2, w_rcp, lo, hi);

            cudarf::Vec2f P0 = to_vec2f(v0 * w_rcp.x);
            cudarf::Vec2f P1 = to_vec2f(v1 * w_rcp.y);
            cudarf::Vec2f P2 = to_vec2f(v2 * w_rcp.z);

            float2 screen = make_float2((float)pipe->windowWidth, (float)pipe->windowHeight);

            cudarf::Vec2f sP0 = 0.5f * screen * (make_float2(P0.x, P0.y) + 1.0f);
            cudarf::Vec2f sP1 = 0.5f * screen * (make_float2(P1.x, P1.y) + 1.0f);
            cudarf::Vec2f sP2 = 0.5f * screen * (make_float2(P2.x, P2.y) + 1.0f);

            float area_f = edge_function(sP0, sP1, sP2);

            if (edge_function(p0, p1, p2) >> (CUDARF_SUBPIXEL_LOG2 - 1) == 0) {return;}

            if (pipe->withFaceCulling && area_f <= 0.0f) {
                return;
            }

            // TODO: implement with fixed-value triangle coords
            // cull triangle if all vertices are outside guard band
            // float loxy = ::min(lo.x, lo.y);
            // float hixy = ::max(hi.x, hi.y);
            // float aabb_limit = (1 << (CUDARF_MAXVIEWPORT_LOG2 + CUDARF_SUBPIXEL_LOG2)) - 1;
            // if (loxy >= CUDARF_VPCOORD_MIN && hixy <= CUDARF_VPCOORD_MAX && hixy - loxy <= aabb_limit)
            {
                // TODO: implement triangle clipping
                pipe->triSubtris[i] = 1;

                setup_triangle<TShaderType, TWithTex>(
                    pipe,
                    &pipe->tris[i],
                    vtx[vidx.x].pos_world,
                    vtx[vidx.y].pos_world,
                    vtx[vidx.z].pos_world,
                    v0, v1, v2,
#ifdef WITH_TAA
                    vtx[vidx.x].pos_ss_hist,
                    vtx[vidx.y].pos_ss_hist,
                    vtx[vidx.z].pos_ss_hist,
#endif
                    vtx[vidx.x].nor,
                    vtx[vidx.y].nor,
                    vtx[vidx.z].nor,
                    vtx[vidx.x].tex,
                    vtx[vidx.y].tex,
                    vtx[vidx.z].tex,
                    vtx[vidx.x].col,
                    vtx[vidx.y].col,
                    vtx[vidx.z].col,
                    sP0, sP1, sP2,
                    w_rcp,
                    1.0 / area_f,
                    pipe->drawPacketMaterials[drawPacketId]);
                return;
            }
        }
    }
}

#endif
