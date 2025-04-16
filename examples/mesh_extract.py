import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import tqdm
from scipy.spatial.transform import Rotation
import warnings

import open3d as o3d
import open3d.core as o3c
import trimesh
from skimage.measure import marching_cubes
from chamfer_distance import ChamferDistance

from gsplat.rendering import rasterization
from gsplat.cuda._torch_impl import (
    gaussian_to_ellipse,
    signed_distance_knn,
    filter_scales
)

class MeshExtract:
    def __init__(
        self, 
        data_dir: str,
        result_dir: str,
        ckpt: str,
        gt_mesh_dir: str,
        #unique_indices: torch.Tensor,
        sdf_loss: bool = False,
        replica: bool = False,
        alpha_threshold: float = 0.0,
        step: int = 29999,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.ckpt = ckpt
        self.sdf_loss = sdf_loss
        self.gt_mesh_dir = gt_mesh_dir
        #self.unique_indices = unique_indices
        self.alpha_threshold = alpha_threshold
        self.step = step

        #self.load_points()
        self.load_splat()
        
        # if replica:
        #     self.load_cameras_replica()
        # else:
        #     self.load_cameras_blender()

    def load_points(self):
        points_path = os.path.join(self.data_dir, "ptcloud.ply")
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"Points file not found: {points_path}")
        
        points = o3d.io.read_point_cloud(points_path)
        points = np.asarray(points.points)
        
        points = torch.tensor(points, device=self.device, dtype=torch.float32)
        self.points = points

    def load_splat(self):
        ckpt_path = os.path.join(self.result_dir, self.ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=self.device)["splats"]

        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
        
        self.means = torch.cat(means, dim=0)
        self.quats = torch.cat(quats, dim=0)
        self.scales = torch.cat(scales, dim=0)
        self.opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        self.colors = torch.cat([sh0, shN], dim=-2)
        self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
        
        if self.sdf_loss:
            sdf0, sdfN = [], []
            sdf0.append(ckpt["sdf0"])
            sdfN.append(ckpt["sdfN"])
            sdf0 = torch.cat(sdf0, dim=0)
            sdfN = torch.cat(sdfN, dim=0)
            self.sdf_coeffs = torch.cat([sdf0, sdfN], dim=-1)
            print(torch.min(self.sdf_coeffs), torch.max(self.sdf_coeffs))
        else:
            self.sdf_coeffs = torch.zeros((self.means.shape[0], 2**(self.sh_degree+1)), device=self.device)
            
        print(f"Loaded {self.means.shape[0]} splats")
    
    def load_cameras_blender(self):
        cam_path = os.path.join(self.data_dir, "cameras.txt")
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera file not found: {cam_path}")
        
        cam_data = np.loadtxt(cam_path, delimiter=",", comments="#")
        cam_id = cam_data[:, 0]
        positions = cam_data[:, 1:4]
        euler_angles = cam_data[:, 4:7] 
        rotations = Rotation.from_euler('xyz', euler_angles, degrees=False).as_matrix()

        N = positions.shape[0]
        camtoworlds = np.zeros((N, 4, 4))
        camtoworlds[:, :3, :3] = rotations      
        camtoworlds[:, :3, 3] = positions      
        camtoworlds[:, 3, 3] = 1.0 

        # Convert from blender to conventional camera coordinates
        T = np.diag([1, -1, -1, 1])
        camtoworlds = camtoworlds @ T
        
        K = np.array([
            [1111.111111,      0, 400.0],
            [      0,   1111.111111, 400.0],
            [      0,          0,     1.0]
        ])

        H, W = (800, 800)

        self.camtoworlds = camtoworlds
        self.K = K
        self.height = H
        self.width = W
        self.num_cams = N

    def load_cameras_replica(self):
        cam_path = os.path.join(self.data_dir, "traj_w_c.txt")
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera file not found: {cam_path}")
        
        cam_data = np.loadtxt(cam_path, delimiter=" ")
        camtoworlds = cam_data.reshape(-1, 4, 4)
        N = camtoworlds.shape[0]
        (H, W) = (480, 640)
        fu = W / 2
        fv = W / 2
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        K = np.array([
            [fu, 0, cx],
            [0, fv, cy],
            [0, 0, 1]
        ])

        self.camtoworlds = camtoworlds
        self.K = K
        self.height = H
        self.width = W
        self.num_cams = N
        


    def extract_rasterization(
        self,
        voxel_size: float = 0.002,
        alpha_thresh: float = 0.5,      
    ):

        camtoworlds_all = torch.tensor(self.camtoworlds, device=self.device, dtype=torch.float32)
        Ks = torch.tensor(self.K, device=self.device, dtype=torch.float32)
        Ks = Ks.unsqueeze(0).repeat(camtoworlds_all.shape[0], 1, 1)

        color_list = []
        depth_list = []
        
        for i in range(camtoworlds_all.shape[0]):
            viewmat = torch.linalg.inv(camtoworlds_all[i].unsqueeze(0))  # [1, 4, 4]
            
            renders, render_alphas, info = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=viewmat,             # [1, 4, 4]
                Ks=Ks[i].unsqueeze(0),         # [1, 3, 3]
                width=self.width,
                height=self.height,
                sh_degree=self.sh_degree,
                render_mode="RGB+ED",          
                distributed=False,
            )
            render_colors = torch.clamp(renders[..., :3], 0.0, 1.0)  
            render_depths = renders[..., 3:4]                       
            
            valid_mask = render_alphas >= alpha_thresh
            render_depths[~valid_mask] = 0
            

            color_np = render_colors[0].cpu().numpy()    
            depth_np = render_depths[0, ..., 0].cpu().numpy()  
            
            color_list.append(np.ascontiguousarray(color_np))
            depth_list.append(depth_np)
        
        # Open3D voxel block grid for TSDF integration.
        o3d_device = o3d.core.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d_device
        )
        
        # For each view, integrate the corresponding depth and color image.
        for i, (color, depth) in enumerate(zip(color_list, depth_list)):
            depth_img = o3d.t.geometry.Image(depth)
            depth_img = depth_img.to(o3d_device)
            color_img = o3d.t.geometry.Image(color)
            color_img = color_img.to(o3d_device)
            
            intrinsic = Ks[i].cpu().numpy().astype(np.float64)  # ensure float64
            intrinsic = o3d.core.Tensor(intrinsic)

            extrinsic = np.linalg.inv(camtoworlds_all[i].cpu().numpy())
            extrinsic = o3d.core.Tensor(extrinsic.astype(np.float64))
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth_img, 
                intrinsic,
                extrinsic, 
                1.0, 8.0
            )
            vbg.integrate(
                frustum_block_coords, 
                depth_img, 
                color_img,
                intrinsic,
                extrinsic,  
                1.0, 8.0
            )
        
        mesh = vbg.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        output_name = f"rasterize_{self.step}.ply"

        output_path = os.path.join(self.result_dir, output_name)    
        o3d.io.write_triangle_mesh(output_path, mesh.to_legacy())
        print("Rasterization extraction complete!")

    def extract_marching_cubes(
            self,
            n = 500,
            batch_size = 100000
    ):
        print(torch.max(self.scales))
        print(torch.max(self.sdf_coeffs))
        means, quats, scales, opacities, sdf_coeffs = filter_scales(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            sdf_coeffs=self.sdf_coeffs,
            alpha_thresh=0.05,
        )



        r_a, r_b, axes_a, axes_b, _ = gaussian_to_ellipse(
            means, 
            quats, 
            scales, 
            opacities
        )


        print(f"Using {means.shape[0]} Gaussians for sdf computations")
        
        offset = torch.max(r_b) + 0.01

        start_x = torch.min(means[:, 0]) - offset
        start_y = torch.min(means[:, 1]) - offset
        start_z = torch.min(means[:, 2]) - offset
        end_x = torch.max(means[:, 0]) + offset
        end_y = torch.max(means[:, 1]) + offset
        end_z = torch.max(means[:, 2]) + offset
        x = torch.linspace(start_x, end_x, n, device=self.device)
        y = torch.linspace(start_y, end_y, n, device=self.device)
        z = torch.linspace(start_z, end_z, n, device=self.device)

        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)


        sdf = []
        total_batches = (grid.shape[0] + batch_size - 1) // batch_size
        for i in tqdm.tqdm(range(0, grid.shape[0], batch_size), total=total_batches, desc="Computing SDF"):
            sdf_batch, _ = signed_distance_knn(
                pos=grid[i:i+batch_size], 
                means=means,
                r_a=r_a, 
                r_b=r_b, 
                axes_a=axes_a, 
                axes_b=axes_b,
                sdf_coeffs=sdf_coeffs,
                sh_degree=self.sh_degree,
                max_threshold=None
            )
            sdf.append(sdf_batch)
        
        sdf = torch.cat(sdf, dim=0)
        sdf = sdf.flatten()
        sdf = sdf.reshape(n, n, n)
        sdf_np = sdf.cpu().numpy()

        dx = (end_x - start_x) / (n - 1)
        dy = (end_y - start_y) / (n - 1)
        dz = (end_z - start_z) / (n - 1)

        dx, dy, dz = dx.cpu().numpy(), dy.cpu().numpy(), dz.cpu().numpy()

        try:
            verts, faces, normals, values = marching_cubes(sdf_np, level=0, spacing=(dx, dy, dz))
        except ValueError:
            warnings.warn("Marching Cubes extraction failed. increasing level set to 0.01")
            verts, faces, normals, values = marching_cubes(sdf_np, level=0.01, spacing=(dx, dy, dz))

        verts += np.array([start_x.cpu(), start_y.cpu(), start_z.cpu()])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        if normals.shape[0] == verts.shape[0]:
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            mesh.compute_vertex_normals()
            
        output_name = f"marching_cubes_{self.step}.ply"
            
        output_path = os.path.join(self.result_dir, output_name)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print("Marching Cubes extraction complete!")
        
    def compute_chamfer_distance(
            self,
            n_samples = 2500000
        ):
        def sample_mesh(m, n):
            vpos, _ = trimesh.sample.sample_surface(m, n)
            return torch.tensor(vpos, dtype=torch.float32, device="cuda")

        def as_mesh(scene_or_mesh):
            if isinstance(scene_or_mesh, trimesh.Scene):
                assert len(scene_or_mesh.geometry) > 0
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()))
            else:
                assert isinstance(scene_or_mesh, trimesh.Trimesh)
                mesh = scene_or_mesh
            return mesh

        gt_path = self.gt_mesh_dir
        mcubes_path = os.path.join(self.result_dir, f"marching_cubes_{self.step}.ply")
        rasterize_path = os.path.join(self.result_dir, f"rasterize_{self.step}.ply")

        gt_mesh = as_mesh(trimesh.load(gt_path))
        mcubes_mesh = as_mesh(trimesh.load(mcubes_path))
        rasterize_mesh = as_mesh(trimesh.load(rasterize_path))
        
        vpos_gt = sample_mesh(gt_mesh, n_samples).unsqueeze(0)
        vpos_mcubes = sample_mesh(mcubes_mesh, n_samples).unsqueeze(0)
        vpos_rasterize = sample_mesh(rasterize_mesh, n_samples).unsqueeze(0)

        chamfer_distance = ChamferDistance()

        dist1_mcubes, dist2_mcubes, _, _ = chamfer_distance(vpos_gt, vpos_mcubes)
        dist1_rasterize, dist2_rasterize, _, _ = chamfer_distance(vpos_gt, vpos_rasterize)
        
        loss_mcubes = (torch.mean(dist1_mcubes) + torch.mean(dist2_mcubes))
        loss_rasterize = (torch.mean(dist1_rasterize) + torch.mean(dist2_rasterize))

        output_name = f"chamfer_distance_{self.step}.txt"
        output_path = os.path.join(self.result_dir, output_name)
        with open(output_path, "w") as f:
            f.write(f"Marching Cubes: {loss_mcubes.item():.9f}\n")
            f.write(f"Rasterization: {loss_rasterize.item():.9f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/monkey_depth")
    parser.add_argument("--result_dir", type=str, default="results/dtu_test_sdf")
    parser.add_argument("--ckpt", type=str, default="ckpts/ckpt_6999_rank0.pt")
    parser.add_argument("--gt_mesh_dir", type=str, default="results/meshes/monkey.ply")
    parser.add_argument("--sdf_loss", type=bool, default=True)
    parser.add_argument("--replica", type=bool, default=False)
    parser.add_argument("--alpha_threshold", type=float, default=0.0)
    parser.add_argument("--step", type=int, default=6999)
    args = parser.parse_args()

    mesh_extract = MeshExtract(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        ckpt=args.ckpt,
        gt_mesh_dir=args.gt_mesh_dir,
        sdf_loss=args.sdf_loss,
        replica=args.replica,
        alpha_threshold=args.alpha_threshold,
        step=args.step,
    )

    #mesh_extract.extract_rasterization()
    mesh_extract.extract_marching_cubes()
    #mesh_extract.compute_chamfer_distance()