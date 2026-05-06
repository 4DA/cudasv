namespace cudarf
{
struct Barycentric
{
    glm::vec3 lambda;
    glm::vec3 ddx;
    glm::vec3 ddy;
};
}

// most straighforward approach: compute barycentric coordinates, given triangle
// vertices in 3DP(NDC) space and raster coordinates
// frag - fragment coordinates in NDC space
__device__ __inline__ cudarf::Vec3f compute_bary_affine(const cudarf::rast::Triangle &tri, const cudarf::Vec2f &frag)
{
    return tri.area_rcp * cudarf::Vec3f {
        edge_function(tri.sP1, tri.sP2, frag),
        edge_function(tri.sP2, tri.sP0, frag),
        edge_function(tri.sP0, tri.sP1, frag)
    };
}

// use simplified calcluation: λ2 = 1 - λ0 - λ1
// should not be used for coverage testing
__device__ __inline__ cudarf::Vec3f compute_bary_affine2(const cudarf::rast::Triangle &tri, const cudarf::Vec2f &frag)
{
    float l0 = tri.area_rcp * edge_function(tri.sP1, tri.sP2, frag);
    float l1 = tri.area_rcp * edge_function(tri.sP2, tri.sP0, frag);
    float l2 = 1.0f - l0 - l1;

    return {l0, l1, l2};
}

__device__ __inline__ cudarf::Vec3f compute_bary_persp(const cudarf::Vec3f &bary, const cudarf::Vec3f &tri_w_rcp)
{
    float w_rcp = dot(bary, tri_w_rcp);
    cudarf::Vec3f baryPersp = 1.0f / w_rcp * bary * tri_w_rcp;
    return baryPersp;
}

__device__
glm::vec3 rcp(glm::vec3 src)
{
    return glm::vec3(1.0 / src.x, 1.0 / src.y, 1.0 / src.z);
}


// source:
// https://github.com/ConfettiFX/The-Forge/blob/2d453f376ef278f66f97cbaf36c0d12e4361e275/Examples_3/Aura/src/Shaders/FSL/visibilityBuffer_shade.frag.fsl#L33
//
// Compact math structure:
//
// 1. Project clip-space vertices into NDC:
//      n_i = P_i.xy / P_i.w
//      q_i = 1 / P_i.w
//
// 2. Affine barycentric terms over screen space are linear in x,y, so their
//    gradients are constant across the triangle:
//      d(lambda_i * q_i)/dx, d(lambda_i * q_i)/dy
//
//    This is what ret.ddx / ret.ddy store component-wise for i in {0,1,2}.
//
// 3. The interpolated reciprocal depth is:
//      q(x, y) = sum_i lambda_i(x, y) * q_i
//
//    and the perspective-correct barycentrics are:
//      lambda_i(x, y) = (lambda_i(x, y) * q_i) / q(x, y)
//
// 4. The final derivative projection step converts the raw affine gradients in
//    the lambda/w domain into perspective-correct per-pixel barycentric
//    gradients that can be used for later attribute differentials.

__device__
cudarf::Barycentric compute_bary_persp_deriv(glm::vec4 P0, glm::vec4 P1, glm::vec4 P2, glm::vec2 ndc, glm::vec2 winSize)
{
    cudarf::Barycentric ret;

    // Perspective-correct interpolation works in 1/w, so convert the clip-space
    // vertex w components into reciprocal form once up front.
    glm::vec3 invW = rcp(glm::vec3(P0.w, P1.w, P2.w));

    // Project the clip-space triangle into NDC. The sample position is also in
    // NDC, so all later barycentric reconstruction happens in this space.
    glm::vec2 ndc0 = glm::vec2(glm::vec2(P0) * invW.x);
    glm::vec2 ndc1 = glm::vec2(glm::vec2(P1) * invW.y);
    glm::vec2 ndc2 = glm::vec2(glm::vec2(P2) * invW.z);

    // Invert the 2x2 screen-space edge matrix once so barycentric gradients can
    // be evaluated from NDC deltas instead of solving the triangle per sample.
    float invDet = 1.0 / glm::determinant(glm::mat2(ndc2 - ndc1, ndc0 - ndc1));

    // These are the per-vertex barycentric gradients multiplied by 1/w, which
    // is the form needed for perspective-correct interpolation.
    ret.ddx = glm::vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.ddy = glm::vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;

    // Summing the per-vertex terms gives the screen-space gradient of the
    // interpolated reciprocal w.
    float ddxSum = glm::dot(ret.ddx, glm::vec3(1.0, 1.0, 1.0));
    float ddySum = glm::dot(ret.ddy, glm::vec3(1.0, 1.0, 1.0));

    glm::vec2 deltaVec = ndc - ndc0;

    // Evaluate 1/w at the sample and invert it back to w for the final
    // perspective-correct barycentric reconstruction.
    float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;
    float interpW = 1.0 / interpInvW;

    // Reconstruct the perspective-correct barycentrics at this sample.
    ret.lambda.x = interpW * (invW[0] + deltaVec.x*ret.ddx.x + deltaVec.y*ret.ddy.x);
    ret.lambda.y = interpW * (0.0f    + deltaVec.x*ret.ddx.y + deltaVec.y*ret.ddy.y);
    ret.lambda.z = interpW * (0.0f    + deltaVec.x*ret.ddx.z + deltaVec.y*ret.ddy.z);

    // to compute screen-space barycentrics directly
    // vec2 dndc0 = ndc - ndc0;
    // vec2 dndc1 = ndc - ndc1;
    // vec2 dndc2 = ndc - ndc2;
    // const vec2 lambda_yz = vec2(determinant(mat2(dndc2, dndc0)), determinant(mat2(dndc0, dndc1))) * invDet;
    // const vec3 lambda = vec3(1.f - lambda_yz[0] - lambda_yz[1], lambda_yz);
    // ret.lambda = lambda;

    // Convert the gradients from NDC units into pixel units so later derivative
    // use lines up with screen-space neighborhoods.
    ret.ddx *= (2.0f/winSize.x);
    ret.ddy *= (2.0f/winSize.y);
    ddxSum  *= (2.0f/winSize.x);
    ddySum  *= (2.0f/winSize.y);

    // This part fixes the derivatives error happening for the projected triangles.
    // Instead of calculating the derivatives constantly across the 2D triangle we use a projected version
    // of the gradients, this is more accurate and closely matches GPU raster behavior.
    // Final gradient equation: ddx = (((lambda/w) + ddx) / (w+|ddx|)) - lambda

    // Evaluate neighboring reciprocal-w values used to project the raw
    // barycentric gradients onto the perspective-correct surface.
    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);

    // Project the raw gradients through perspective so their behavior matches
    // hardware-style post-projection attribute interpolation.
    ret.ddx = interpW_ddx*(ret.lambda*interpInvW + ret.ddx) - ret.lambda;
    ret.ddy = interpW_ddy*(ret.lambda*interpInvW + ret.ddy) - ret.lambda;

    return ret;
}

