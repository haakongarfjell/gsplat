#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "Ops.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_GAUSSIANS_PER_RAY 16384

namespace gsplat {

template <typename T>
__global__ void sphere_trace_kernel(
    const uint32_t H,
    const uint32_t W,
    const uint32_t N,
    const T* __restrict__ means,
    const T* __restrict__ quats,
    const T* __restrict__ scales,
    const T* __restrict__ colors,
    const T* __restrict__ origins,
    const T* __restrict__ directions,
    const T* __restrict__ r_a,
    const T* __restrict__ r_b,
    const T* __restrict__ axes_a,
    const T* __restrict__ axes_b,
    const int max_steps,
    const T min_hit_distance,
    const T max_trace_distance,
    T* __restrict__ total_distance,
    T* __restrict__ hit_color
)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= H || j >= W) return;

    int idx = i * W + j;
    float3 ro = make_float3(origins[idx * 3], origins[idx * 3 + 1], origins[idx * 3 + 2]);
    float3 rd = make_float3(directions[idx * 3], directions[idx * 3 + 1], directions[idx * 3 + 2]);

    T t = 0.0;
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    T final_distance = max_trace_distance;

    // Step 1: Cull Gaussians for this ray (in registers/local mem)
    int valid_count = 0;

    for (int step = 0; step < max_steps; ++step) {
        float3 pos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);

        T min_dist = max_trace_distance;
        int closest_idx = -1;

        for (int n = 0; n < N; ++n) {
            float3 mean = make_float3(means[n * 3], means[n * 3 + 1], means[n * 3 + 2]);
            float3 axesA = make_float3(axes_a[n * 3], axes_a[n * 3 + 1], axes_a[n * 3 + 2]);
            float3 axesB = make_float3(axes_b[n * 3], axes_b[n * 3 + 1], axes_b[n * 3 + 2]);
            T rA = r_a[n];
            T rB = r_b[n];

            float3 diff = make_float3(pos.x - mean.x, pos.y - mean.y, pos.z - mean.z);
            T proj_a = diff.x * axesA.x + diff.y * axesA.y + diff.z * axesA.z;
            T proj_b = diff.x * axesB.x + diff.y * axesB.y + diff.z * axesB.z;

            float3 proj_point = make_float3(
                proj_a * axesA.x + proj_b * axesB.x,
                proj_a * axesA.y + proj_b * axesB.y,
                proj_a * axesA.z + proj_b * axesB.z
            );

            T scaled_a = proj_a / rA;
            T scaled_b = proj_b / rB;
            T scale_factor = sqrtf(scaled_a * scaled_a + scaled_b * scaled_b);

            float3 closest_point_boundary = make_float3(
                (proj_a / scale_factor) * axesA.x + (proj_b / scale_factor) * axesB.x,
                (proj_a / scale_factor) * axesA.y + (proj_b / scale_factor) * axesB.y,
                (proj_a / scale_factor) * axesA.z + (proj_b / scale_factor) * axesB.z
            );

            T dist_proj = sqrtf((diff.x - proj_point.x) * (diff.x - proj_point.x) +
                                (diff.y - proj_point.y) * (diff.y - proj_point.y) +
                                (diff.z - proj_point.z) * (diff.z - proj_point.z));

            T dist_boundary = sqrtf((diff.x - closest_point_boundary.x) * (diff.x - closest_point_boundary.x) +
                                    (diff.y - closest_point_boundary.y) * (diff.y - closest_point_boundary.y) +
                                    (diff.z - closest_point_boundary.z) * (diff.z - closest_point_boundary.z));

            bool inside_mask = (scaled_a * scaled_a + scaled_b * scaled_b) <= 1.0f;
            T dist = inside_mask ? dist_proj : dist_boundary;

            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = n;
            }
        }

        if (min_dist < min_hit_distance) {
            final_distance = t;
            final_color = make_float3(colors[closest_idx * 3], colors[closest_idx * 3 + 1], colors[closest_idx * 3 + 2]);
            break;
        }

        t += min_dist;
        if (t > max_trace_distance) break;
    }

    total_distance[idx] = final_distance;
    hit_color[idx * 3] = final_color.x;
    hit_color[idx * 3 + 1] = final_color.y;
    hit_color[idx * 3 + 2] = final_color.z;
}

std::tuple<at::Tensor, at::Tensor> sphere_trace(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &colors,
    const at::Tensor &origins,
    const at::Tensor &directions,
    const at::Tensor &r_a,
    const at::Tensor &r_b,
    const at::Tensor &axes_a,
    const at::Tensor &axes_b,
    const int max_steps,
    const float min_hit_distance,
    const float max_trace_distance
) {

    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(origins);
    CHECK_INPUT(directions);
    CHECK_INPUT(r_a);
    CHECK_INPUT(r_b);
    CHECK_INPUT(axes_a);
    CHECK_INPUT(axes_b);

    uint32_t H = origins.size(0);
    uint32_t W = origins.size(1);
    uint32_t N = means.size(0);

    auto total_distance = at::zeros({H, W}, means.options());
    auto hit_color = at::zeros({H, W, 3}, means.options());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        means.scalar_type(), "sphere_trace_kernel", ([&] {
            sphere_trace_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                H, W, N,
                means.data_ptr<scalar_t>(),
                quats.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                colors.data_ptr<scalar_t>(),
                origins.data_ptr<scalar_t>(),
                directions.data_ptr<scalar_t>(),
                r_a.data_ptr<scalar_t>(),
                r_b.data_ptr<scalar_t>(),
                axes_a.data_ptr<scalar_t>(),
                axes_b.data_ptr<scalar_t>(),
                max_steps,
                min_hit_distance,
                max_trace_distance,
                total_distance.data_ptr<scalar_t>(),
                hit_color.data_ptr<scalar_t>()
            );
        }));

    return std::make_tuple(hit_color, total_distance);
}

} // namespace gsplat
