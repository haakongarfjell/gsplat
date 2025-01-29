#include "bindings.h"
#include "helpers.cuh"
#include "transform.cuh"
#include "2dgs.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

/****************************************************************************
 * Projection of Gaussians (Batched) Backward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_packed_bwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4]
    const T *__restrict__ scales,   // [N, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const T *__restrict__ ray_transforms,     // [nnz, 3]
    const T *__restrict__ v_means2d,          // [nnz, 2]
    const T *__restrict__ v_depths,           // [nnz]
    const T *__restrict__ v_normals,          // [nnz, 3]
    const bool sparse_grad,                   // whether the outputs are in COO format [nnz, ...]
    T *__restrict__ v_ray_transforms,
    T *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    T *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    T *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    T *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) {
        return;
    }

    const int64_t cid = camera_ids[idx];   // Camera ID
    const int64_t gid = gaussian_ids[idx]; // Gaussian ID

    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;
    v_means2d += idx * 2;
    v_normals += idx * 3;
    v_depths += idx;
    v_ray_transforms += idx * 9;

    mat3<T> R = mat3<T>(
        viewmats[0], viewmats[4], viewmats[8],
        viewmats[1], viewmats[5], viewmats[9],
        viewmats[2], viewmats[6], viewmats[10]
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);

    vec4<T> quat = glm::make_vec4(quats + gid * 4);
    vec2<T> scale = glm::make_vec2(scales + gid * 3);

    mat3<T> P = mat3<T>(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3<T> _v_ray_transforms = mat3<T>(
        v_ray_transforms[0], v_ray_transforms[1], v_ray_transforms[2],
        v_ray_transforms[3], v_ray_transforms[4], v_ray_transforms[5],
        v_ray_transforms[6], v_ray_transforms[7], v_ray_transforms[8]
    );
    _v_ray_transforms[2][2] += v_depths[0];

    vec3<T> v_normal = glm::make_vec3(v_normals);

    vec3<T> v_mean(0.f);
    vec2<T> v_scale(0.f);
    vec4<T> v_quat(0.f);

    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean
    );

    if (sparse_grad) {
        // Write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
            v_means[0] = v_mean.x;
            v_means[1] = v_mean.y;
            v_means[2] = v_mean.z;
        }

        if (v_quats != nullptr) {
            v_quats += idx * 4;
            v_quats[0] = v_quat[0];
            v_quats[1] = v_quat[1];
            v_quats[2] = v_quat[2];
            v_quats[3] = v_quat[3];
        }

        if (v_scales != nullptr) {
            v_scales += idx * 3;
            v_scales[0] = v_scale.x;
            v_scales[1] = v_scale.y;
        }
    } else {
        if (v_means != nullptr) {
            gpuAtomicAdd(v_means + gid * 3 + 0, v_mean.x);
            gpuAtomicAdd(v_means + gid * 3 + 1, v_mean.y);
            gpuAtomicAdd(v_means + gid * 3 + 2, v_mean.z);
        }

        if (v_quats != nullptr) {
            gpuAtomicAdd(v_quats + gid * 4 + 0, v_quat[0]);
            gpuAtomicAdd(v_quats + gid * 4 + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + gid * 4 + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + gid * 4 + 3, v_quat[3]);
        }

        if (v_scales != nullptr) {
            gpuAtomicAdd(v_scales + gid * 3 + 0, v_scale.x);
            gpuAtomicAdd(v_scales + gid * 3 + 1, v_scale.y);
        }

        if (v_viewmats != nullptr) {
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; ++i) {
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; ++j) {
                    gpuAtomicAdd(v_viewmats + cid * 16 + i * 4 + j, R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + cid * 16 + i * 4 + 3, t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_packed_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const torch::Tensor &camera_ids,   // [nnz]
    const torch::Tensor &gaussian_ids, // [nnz]
    const torch::Tensor &ray_transforms,       // [nnz, 3, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [nnz, 2]
    const torch::Tensor &v_depths,  // [nnz]
    const torch::Tensor &v_ray_transforms,  // [nnz, 3, 3]
    const torch::Tensor &v_normals, // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(camera_ids);
    GSPLAT_CHECK_INPUT(gaussian_ids);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ray_transforms);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());

        v_quats = torch::zeros({nnz, 4}, quats.options());
        v_scales = torch::zeros({nnz, 3}, scales.options());

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }

    } else {
        v_means = torch::zeros_like(means);

        v_quats = torch::zeros_like(quats);
        v_scales = torch::zeros_like(scales);

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }
    if (nnz) {

        fully_fused_projection_packed_bwd_2dgs_kernel<float>
            <<<(nnz + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                nnz,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                ray_transforms.data_ptr<float>(),
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                sparse_grad,
                v_ray_transforms.data_ptr<float>(),
                v_means.data_ptr<float>(),
                v_quats.data_ptr<float>(),
                v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat