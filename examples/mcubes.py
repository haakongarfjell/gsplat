from gsplat.cuda._torch_impl import (
    generate_rays,
    gaussian_to_ellipse,
    gaussian_to_ellipse_old,
    calculate_depth,
    generate_depth_image,
    outer_ellipsoid,
    _quat_scale_to_covar_preci,
    signed_distance,
    signed_distance_knn
)
import torch
import torch.nn.functional as F
import math
from skimage.measure import marching_cubes
import trimesh
import numpy as np


ckpt_path = "/home/admin/haakon/gsplat/examples/results/monkey_depth_harmonics1/ckpts/ckpt_29999_rank0.pt"
means, quats, scales, opacities, sh0, shN, sdf0, sdfN = [], [], [], [], [], [], [], []

ckpt = torch.load(ckpt_path, map_location="cuda")["splats"]
means.append(ckpt["means"])
quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
scales.append(torch.exp(ckpt["scales"]))
opacities.append(torch.sigmoid(ckpt["opacities"]))
sh0.append(ckpt["sh0"])
shN.append(ckpt["shN"])
sdf0.append(ckpt["sdf0"])
sdfN.append(ckpt["sdfN"])

means = torch.cat(means, dim=0)
quats = torch.cat(quats, dim=0)
scales = torch.cat(scales, dim=0)
opacities = torch.cat(opacities, dim=0)
sh0 = torch.cat(sh0, dim=0)
shN = torch.cat(shN, dim=0)
colors = torch.cat([sh0, shN], dim=-2)
sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
sdf0 = torch.cat(sdf0, dim=0)
sdfN = torch.cat(sdfN, dim=0)
sdf_coeffs = torch.cat([sdf0, sdfN], dim=-1)
#sdf_coeffs = torch.zeros((means.shape[0], 16), device="cuda")
r_a, r_b, axes_a, axes_b = gaussian_to_ellipse(means, quats, scales, opacities)

print(means.shape)
print(sdf_coeffs.shape)

# mask = (
#     ~torch.isnan(means).any(dim=1) &
#     ~torch.isnan(quats).any(dim=1) &
#     ~torch.isnan(scales).any(dim=1) &
#     ~torch.isnan(opacities).any(dim=0) &
#     ~torch.isnan(sdf_coeffs).any(dim=1)
# )

# means       = means[mask]
# quats       = quats[mask]
# scales      = scales[mask]
# opacities   = opacities[mask]
# sdf_coeffs  = sdf_coeffs[mask]
# print(means.shape)
# print(sdf_coeffs.shape)

print(torch.min(means))
print(torch.max(means))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 500
x = torch.linspace(-1.5, 1.5, n, device=device)
y = torch.linspace(-1.5, 1.5, n, device=device)
z = torch.linspace(-1.5, 1.5, n, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

# Reshape each coordinate grid to a 1D tensor and stack to get a (N, 3) tensor
points = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)

print(points.shape)  # Expected: torch.Size([n**3, 3])

batch_size = 100000
num_points = points.shape[0]



sdf = []

for i in range(0, num_points, batch_size):
    print(i)
    batch = points[i : i + batch_size]

    sdf_batch, _ = signed_distance_knn(
        pos=batch, 
        means=means, 
        r_a=r_a, 
        r_b=r_b, 
        axes_a=axes_a, 
        axes_b=axes_b, 
        sdf_coeffs=sdf_coeffs,
        sh_degree=1,
    )

    sdf.append(sdf_batch)

sdf = torch.cat(sdf, dim=0)
sdf = sdf.flatten()

sdf_volume = sdf.reshape(n, n, n)

sdf_np = sdf_volume.cpu().numpy()

verts, faces, normals, values = marching_cubes(sdf_np, level=0)

print("Extracted mesh vertices shape:", verts.shape)
print("Extracted mesh faces shape:", faces.shape)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# Export to an OBJ file
mesh.export('mcubes_depth_harmonics1.obj')

import open3d as o3d

# Assuming you have your marching cubes results:
# verts, faces, normals, values = marching_cubes(sdf_np, level=0.01)

# Create an Open3D TriangleMesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# If normals are available and correctly sized, set them; otherwise, compute them.
if normals.shape[0] == verts.shape[0]:
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
else:
    print("compute")
    mesh.compute_vertex_normals()

# Optionally, if you want to include color information, you can set:
# mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# Save the mesh as a PLY file, matching the format used in your provided code.
o3d.io.write_triangle_mesh("mcubes_depth_harmonics1.ply", mesh)
print("Mesh saved as mcubes_hessian.ply")