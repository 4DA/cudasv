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
// Computes perspective-correct barycentric coordinates and their screen-space
// derivatives at a sample point inside a projected triangle. Used in
// visibility-buffer shading to reconstruct attributes (and their gradients
// for mip selection / anisotropic filtering) without an interpolator.
//
// -----------------------------------------------------------------------------
// Notation used in this file
// -----------------------------------------------------------------------------
//   P_i      : clip-space vertex i (xyzw)                 [i in {0,1,2}]
//   w_i      : clip-space w of vertex i                   (per-vertex constant)
//   q_i      : 1 / w_i                                    (per-vertex constant)
//   ndc_i    : (P_i.x / w_i, P_i.y / w_i)                 (vertex in NDC)
//
//   p        : the sample position (in NDC)
//   w(p)     : clip-space w of the 3D point under p (varies across triangle)
//   q(p)     : 1 / w(p)
//
//   lambda_i : SCREEN-SPACE barycentrics of the projected 2D triangle.
//              Linear in (x, y), so their gradients are constants over the
//              triangle. Sum to 1.
//   alpha_i  : PERSPECTIVE-CORRECT (object-space) barycentrics of the actual
//              3D point under p. These are what you'd dot with per-vertex
//              attributes to interpolate correctly. Rational in (x, y).
//
// -----------------------------------------------------------------------------
// Key identities
// -----------------------------------------------------------------------------
//   q(p)     = sum_i lambda_i(p) * q_i             (1/w is linear in screen)
//   alpha_i  = (lambda_i / w_i) / q(p)
//            = (lambda_i / w_i) * w(p)
//
// The "numerator field" lambda_i / w_i is linear in screen position, so its
// gradient is a constant. That's the form we step in. alpha_i itself is
// rational and cannot be stepped with a constant gradient -- hence the
// perspective-correction dance at the end.
//
// -----------------------------------------------------------------------------
// What the output struct holds (after the function returns)
// -----------------------------------------------------------------------------
//   ret.lambda : alpha_i at the sample (perspective-correct barycentrics).
//                NB: the field is named "lambda" by convention in this codebase
//                even though mathematically it stores alpha_i.
//   ret.ddx    : forward finite difference  alpha_i(x+1, y) - alpha_i(x, y).
//   ret.ddy    : forward finite difference  alpha_i(x, y+1) - alpha_i(x, y).
//                Both match the semantics of GPU dFdx / dFdy.
__device__
cudarf::Barycentric compute_bary_persp_deriv(glm::vec4 P0, glm::vec4 P1, glm::vec4 P2, glm::vec2 ndc, glm::vec2 winSize)
{
    cudarf::Barycentric ret;

    // ---- Step 1: project the triangle to NDC -------------------------------
    // Compute q_i = 1/w_i once. Perspective-correct math is cleanest in
    // reciprocal-w form because 1/w is linear in screen space, while w isn't.
    glm::vec3 invW = rcp(glm::vec3(P0.w, P1.w, P2.w));
    // ndc_i = P_i.xy * q_i. The sample 'ndc' is already in this space, so
    // everything from here on lives in NDC.
    glm::vec2 ndc0 = glm::vec2(glm::vec2(P0) * invW.x);
    glm::vec2 ndc1 = glm::vec2(glm::vec2(P1) * invW.y);
    glm::vec2 ndc2 = glm::vec2(glm::vec2(P2) * invW.z);

    // ---- Step 2: per-vertex screen-space gradients of (lambda_i / w_i) -----
    // 1 / det = 1 / (2 * signed area of the NDC triangle). Same denominator
    // appears in every barycentric formula, so factor it out once.
    float invDet = 1.0 / glm::determinant(glm::mat2(ndc2 - ndc1, ndc0 - ndc1));

    // For a 2D triangle, d(lambda_i)/dx and d(lambda_i)/dy are constants given
    // by closed-form expressions in the vertex coordinates:
    //     d(lambda_0)/dx = (y_1 - y_2) / det,  etc.
    // Multiplying component-wise by invW = (q_0, q_1, q_2) gives the gradient
    // of the "numerator field" lambda_i / w_i, which is what perspective-
    // correct interpolation actually needs. These gradients are constant
    // across the triangle (the field is linear in screen position).
    ret.ddx = glm::vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.ddy = glm::vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;

    // ---- Step 3: gradient of q(p) = 1/w at the sample ----------------------
    // q(p) = sum_i lambda_i(p) * q_i, so its screen-space gradient is the
    // sum of the per-vertex gradients we just built:
    //     dq/dx = sum_i d(lambda_i)/dx * q_i = sum_i ret.ddx[i].
    // This collapses the 3-vector to a scalar because q is a single field.
    float ddxSum = glm::dot(ret.ddx, glm::vec3(1.0, 1.0, 1.0));
    float ddySum = glm::dot(ret.ddy, glm::vec3(1.0, 1.0, 1.0));

    // ---- Step 4: evaluate q(p) and w(p) at the sample ----------------------
    // q is a plane in screen space (linear in x, y), so we can evaluate it
    // anywhere from a known point plus the constant gradients. Pick vertex 0
    // as the known point: at v_0, lambda = (1, 0, 0), so q = q_0 = invW[0].
    glm::vec2 deltaVec = ndc - ndc0;

    // Plane interpolation: q(p) = q_0 + dq/dx * delta_x + dq/dy * delta_y.
    float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;

    // Invert to get w(p), needed below to convert numerator -> alpha.
    float interpW = 1.0 / interpInvW;

    // ---- Step 5: build alpha_i at the sample -------------------------------
    // Same plane-interpolation trick, but now applied component-wise to each
    // of the three numerator fields lambda_i / w_i. At vertex 0 their values
    // are:
    //     lambda_0/w_0 |_v0 = 1/w_0 = invW[0]
    //     lambda_1/w_1 |_v0 = 0
    //     lambda_2/w_2 |_v0 = 0
    // Walk along the plane to get lambda_i/w_i at the sample, then multiply
    // by w(p) to get the perspective-correct barycentric:
    //     alpha_i = (lambda_i / w_i) * w(p).
    ret.lambda.x = interpW * (invW[0] + deltaVec.x*ret.ddx.x + deltaVec.y*ret.ddy.x);
    ret.lambda.y = interpW * (0.0f    + deltaVec.x*ret.ddx.y + deltaVec.y*ret.ddy.y);
    ret.lambda.z = interpW * (0.0f    + deltaVec.x*ret.ddx.z + deltaVec.y*ret.ddy.z);

    // Alternative path: compute screen-space lambda_i directly from sub-
    // triangle areas (ratio of signed areas), then perspective-correct
    // afterward. Equivalent result, kept here for reference.
    // vec2 dndc0 = ndc - ndc0;
    // vec2 dndc1 = ndc - ndc1;
    // vec2 dndc2 = ndc - ndc2;
    // const vec2 lambda_yz = vec2(determinant(mat2(dndc2, dndc0)), determinant(mat2(dndc0, dndc1))) * invDet;
    // const vec3 lambda = vec3(1.f - lambda_yz[0] - lambda_yz[1], lambda_yz);
    // ret.lambda = lambda;

    // ---- Step 6: rescale gradients from per-NDC to per-pixel units ---------
    // NDC spans [-1, 1] across winSize pixels, so one pixel step in screen x
    // is 2 / winSize.x in NDC. Chain rule: d/dx_pixel = d/dx_ndc * 2/winSize.x.
    // After this rescale, all gradients are "per pixel," which is what the
    // forward-difference step below expects.
    ret.ddx *= (2.0f/winSize.x);
    ret.ddy *= (2.0f/winSize.y);
    ddxSum  *= (2.0f/winSize.x);
    ddySum  *= (2.0f/winSize.y);

    // ---- Step 7: perspective-correct the gradients -------------------------
    // The raw ret.ddx/ret.ddy are gradients of lambda_i / w_i (the numerator),
    // not of alpha_i itself. Because alpha_i = num / q is rational in screen
    // position, its true derivative is NOT a constant -- you can't just step
    // alpha_i by a fixed gradient. So we form a forward finite difference of
    // alpha_i instead, which matches GPU dFdx/dFdy semantics.
    //
    // Pattern: un-correct -> step linearly in numerator space -> re-correct
    //          -> subtract current value:
    //
    //   alpha_i(x+1, y) = (lambda_i/w_i + d(lambda_i/w_i)/dx)
    //                     -----------------------------------
    //                          ( q(p) + dq/dx )
    //
    //   ret.ddx[i]      = alpha_i(x+1, y) - alpha_i(x, y)     (fwd difference)
    //
    // Note: ret.lambda * interpInvW recovers lambda_i/w_i at the sample
    // (un-corrects), since alpha_i * (1/w(p)) = lambda_i/w_i.
    float interpW_ddx = 1.0f / (interpInvW + ddxSum);  // w(p) at the +x neighbor
    float interpW_ddy = 1.0f / (interpInvW + ddySum);  // w(p) at the +y neighbor
    ret.ddx = interpW_ddx*(ret.lambda*interpInvW + ret.ddx) - ret.lambda;
    ret.ddy = interpW_ddy*(ret.lambda*interpInvW + ret.ddy) - ret.lambda;

    return ret;
}
