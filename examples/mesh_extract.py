import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import tqdm
from scipy.spatial.transform import Rotation
import warnings
from typing import Dict, Any
import open3d as o3d
import open3d.core as o3c
import trimesh
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
from skimage.measure import marching_cubes
from matplotlib import cm
from typing import Tuple

from gsplat.rendering import (
    rasterization,
    rasterization_2dgs
)
from gsplat.cuda._torch_impl import (
    gaussian_to_ellipse,
    signed_distance_knn,
    filter_scales
)


def sample_single_tri(args):
    n1, n2, v1, v2, base = args
    c = np.mgrid[:n1+1, :n2+1].astype(np.float32)
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    pts = np.stack([c[0].ravel(), c[1].ravel()], axis=1)
    mask = pts.sum(axis=1) < 1
    k = pts[mask]
    return (v1 * k[:, :1] + v2 * k[:, 1:] + base).astype(np.float32)

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
        dtu: bool = False,
        refined: bool = False,
        twodgs: bool = False,
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
        self.dtu = dtu
        self.twodgs = twodgs

        #self.load_points()
        self.load_splat()
        
        if replica:
            self.load_cameras_replica()
        elif dtu:
            self.load_cameras_dtu()
        else:
            self.load_cameras_blender()

        if refined:
            self.load_gaussian_ids()


    def load_gaussian_ids(self):
        def prune_id_to_count(
            id_to_count: Dict[Any, int],
            threshold: int = 2
        ) -> Dict[Any, int]:
            return {k: v for k, v in id_to_count.items() if v >= threshold}

        id_path = os.path.join(self.result_dir, f"stats/id_to_count_{self.step}.pt")
        if not os.path.exists(id_path):
            raise FileNotFoundError(f"ID file not found: {id_path}")
        id_to_count = torch.load(id_path)

        id_to_count = prune_id_to_count(id_to_count, threshold=20)

        key_list = list(id_to_count.keys())
        self.gaussian_ids = torch.tensor(key_list, dtype=torch.long, device=self.device)


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
        
    def load_cameras_dtu(self):
        def read_cam(filepath):
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if line.strip() != ""]
            
            if lines[0].lower() != "extrinsic":
                raise ValueError("File format error: The first non-empty line must be 'extrinsic'")
            
            extrinsic = np.array([list(map(float, line.split())) for line in lines[1:5]])
            
            if lines[5].lower() != "intrinsic":
                raise ValueError("File format error: Expected 'intrinsic' after extrinsic matrix data.")
            
            intrinsic = np.array([list(map(float, line.split())) for line in lines[6:9]])
            
            intrinsic[:2] *= 4

            extrinsic[:3, 3] *= 1/200


            extra_params = np.array(list(map(float, lines[9].split())))

            depth_min = extra_params[0] * 1/200
            depth_max = depth_min + extra_params[1] * 192 * 1/200

            extra_params[0] = depth_max
            extra_params[1] = depth_min
            
            return extrinsic, intrinsic, extra_params

        cam_dir = os.path.join(self.data_dir, "Cameras/train")
        if not os.path.exists(cam_dir):
            raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
        
        camtoworlds = []
        K = None

        for image_number in range(0,49):
            cam_name = f"{image_number:08}"
            cam_path = os.path.join(cam_dir, f"{cam_name}_cam.txt")

            w2c, K_m, near_fars = read_cam(cam_path)
            c2w = np.linalg.inv(w2c).reshape(-1, 4, 4)
            camtoworlds.append(c2w)

            K = K_m
            
        camtoworlds = np.concatenate(camtoworlds, axis=0)
        self.camtoworlds = camtoworlds
        self.K = K
        N = camtoworlds.shape[0]
        self.num_cams = N
        (H, W) = (512, 640)
        self.height = H
        self.width = W
        


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
            
            if self.twodgs:
                
                render_colors, render_alphas, render_normals, normals_from_depth, render_distort, render_median, info = rasterization_2dgs(
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
                )
                render_depths = render_median 
                print(f"Render median: {render_median}")
            else:
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
                render_colors = renders[..., :3]  # [1, H, W, 3]
                render_depths = renders[..., 3:4]  # [1, H, W, 1]
            
            valid_mask = render_alphas >= alpha_thresh
            render_depths[~valid_mask] = 0

            
            color_np = render_colors[0].detach().cpu().numpy()    
            depth_np = render_depths[0, ..., 0].detach().cpu().numpy()  
            
            print(np.min(depth_np), np.max(depth_np))
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


        means = self.means[self.gaussian_ids]
        quats = self.quats[self.gaussian_ids]
        scales = self.scales[self.gaussian_ids]
        opacities = self.opacities[self.gaussian_ids]
        sdf_coeffs = self.sdf_coeffs[self.gaussian_ids]

        opacity_threshold = 0.1
        mask = opacities > opacity_threshold
        means = means[mask]
        quats = quats[mask]
        scales = scales[mask]
        opacities = opacities[mask]
        sdf_coeffs = sdf_coeffs[mask]



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

    def extract_sdf_plane(
        self,
        origin: Tuple[float,float,float],    
        normal: Tuple[float,float,float],    
        width: float = 2.0,                 
        height: float = 2.0,               
        n: int = 500,                     
        batch_size: int = 100_000,
        colormap: str = "hot"
    ):
        origin_t = torch.tensor(origin, device=self.device, dtype=torch.float32)
        normal_t = torch.tensor(normal, device=self.device, dtype=torch.float32)
        normal_t = normal_t / normal_t.norm()

        # 2) pick a helper axis
        z_axis = origin_t.new_tensor([0.0, 0.0, 1.0])
        if torch.allclose(normal_t.abs(), z_axis):
            helper = origin_t.new_tensor([0.0, 1.0, 0.0])
        else:
            helper = z_axis

        # 3) build your in-plane axes u, v
        u = torch.cross(normal_t, helper)
        u = u / u.norm()
        v = torch.cross(normal_t, u)

        # 4) now you can safely use `n` as an int
        us = torch.linspace(-width/2, width/2, n,  device=self.device)
        vs = torch.linspace(-height/2, height/2, n, device=self.device)
        U, V = torch.meshgrid(us, vs, indexing="ij")

        pts = origin_t[None] + U.reshape(-1,1)*u[None] + V.reshape(-1,1)*v[None]
        sdf_vals = []
        means      = self.means[self.gaussian_ids]
        r_a, r_b, axes_a, axes_b, _ = gaussian_to_ellipse(
            means, self.quats[self.gaussian_ids],
            self.scales[self.gaussian_ids],
            self.opacities[self.gaussian_ids]
        )
        for i in range(0, pts.shape[0], batch_size):
            sdf_b, _ = signed_distance_knn(
                pos=pts[i:i+batch_size],
                means=means, r_a=r_a, r_b=r_b,
                axes_a=axes_a, axes_b=axes_b,
                sdf_coeffs=self.sdf_coeffs[self.gaussian_ids],
                sh_degree=self.sh_degree
            )
            sdf_vals.append(sdf_b)
        sdf = torch.cat(sdf_vals, dim=0).cpu().numpy()

        vmin, vmax = float(sdf.min()), float(sdf.max())
        normed     = (sdf - vmin) / (vmax - vmin + 1e-12)
        cmap       = cm.get_cmap(colormap)
        colors     = cmap(normed)[:,:3].astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)
        out = os.path.join(self.result_dir, f"sdf_plane_{self.step}.ply")
        o3d.io.write_point_cloud(out, pcd)
        print(f"Wrote plane SDF to {out}")


    
    def sample_mesh(self, mesh: trimesh.Trimesh, density: float) -> np.ndarray:
        verts     = mesh.vertices                     # (V,3) numpy
        faces     = mesh.faces                        # (F,3)
        tri_verts = verts[faces]                      # (F,3,3)
        v1 = tri_verts[:,1] - tri_verts[:,0]          # (F,3)
        v2 = tri_verts[:,2] - tri_verts[:,0]          # (F,3)
        l1 = np.linalg.norm(v1, axis=1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=1, keepdims=True)

        # remove degenerate tris
        mask_valid = (area2.squeeze() > 0)
        v1, v2, tri_verts, l1, l2, area2 = [
            arr[mask_valid] for arr in (v1, v2, tri_verts, l1, l2, area2)
        ]

        # per-tri sample counts
        thr = density * np.sqrt((l1 * l2) / area2)
        n1  = np.floor(l1 / thr).astype(int)
        n2  = np.floor(l2 / thr).astype(int)

        args = [
            (int(n1[i,0]), int(n2[i,0]),
             v1[i], v2[i],
             tri_verts[i,0])
            for i in range(len(n1))
        ]

        # pure-NumPy multiprocessing
        ctx = mp.get_context("spawn")
        with ctx.Pool() as p:
            new_pts = p.map(sample_single_tri, args)

        new_pts = np.vstack(new_pts)                  # (M,3)
        all_pts = np.vstack([verts, new_pts])         # (V+M,3)
        return all_pts
        

    def sample_pointcloud(self, pc: trimesh.points.PointCloud, density: float) -> np.ndarray:
 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pc.vertices, dtype=np.float32))
        # voxel downsample
        down = pcd.voxel_down_sample(voxel_size=density)
        # return as raw numpy
        return np.asarray(down.points, dtype=np.float32)
    
    def compute_chamfer_distance(
        self,
        density: float = 0.2,
        max_dist: float = 20.0
    ):
        def as_mesh(x):
            if isinstance(x, trimesh.Scene):
                return trimesh.util.concatenate(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in x.geometry.values()
                )
            return x

        # load GT
        gt_path = self.gt_mesh_dir
        if self.dtu:
            pc     = trimesh.load(gt_path)
            assert isinstance(pc, trimesh.points.PointCloud)
            print("sample pointcloud")
            gt_pts = self.sample_pointcloud(pc, density)
        else:
            m      = as_mesh(trimesh.load(gt_path))
            gt_pts = self.sample_mesh(m, density)

        # load recs
        mc = as_mesh(trimesh.load(os.path.join(self.result_dir, f"marching_cubes_{self.step}.ply")))
        ra = as_mesh(trimesh.load(os.path.join(self.result_dir, f"rasterize_{self.step}.ply")))

        mc_pts = self.sample_mesh(mc, density)
        ra_pts = self.sample_mesh(ra, density)

        print("chamfer")
        # symmetric Chamfer via sklearn
        def dtu_chamfer(a_pts, b_pts):
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
            nn.fit(b_pts)
            d1, _ = nn.kneighbors(a_pts)
            m1    = d1 < max_dist
            mean1 = d1[m1].mean()
            nn.fit(a_pts)
            d2, _ = nn.kneighbors(b_pts)
            m2    = d2 < max_dist
            mean2 = d2[m2].mean()
            return float((mean1 + mean2) / 2.0)

        loss_mcubes    = dtu_chamfer(mc_pts, gt_pts)
        loss_rasterize = dtu_chamfer(ra_pts, gt_pts)

        # write out
        fn = os.path.join(self.result_dir, f"chamfer_distance_{self.step}.txt")
        with open(fn, "w") as f:
            f.write(f"Marching Cubes: {loss_mcubes:.9f}\n")
            f.write(f"Rasterization:  {loss_rasterize:.9f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DTU/scan24")
    parser.add_argument("--result_dir", type=str, default="results/dtu_sdf_scan24")
    parser.add_argument("--ckpt", type=str, default="ckpts/ckpt_29999_rank0.pt")
    parser.add_argument("--gt_mesh_dir", type=str, default="data/DTU/scan24/depth_dtu_scan24_full.ply")
    parser.add_argument("--sdf_loss", type=bool, default=True)
    parser.add_argument("--replica", type=bool, default=False)
    parser.add_argument("--dtu", type=bool, default=True)
    parser.add_argument("--refined", type=bool, default=True)
    parser.add_argument("--twodgs", type=bool, default=False)
    parser.add_argument("--alpha_threshold", type=float, default=0.0)
    parser.add_argument("--step", type=int, default=29999)
    args = parser.parse_args()

    mesh_extract = MeshExtract(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        ckpt=args.ckpt,
        gt_mesh_dir=args.gt_mesh_dir,
        sdf_loss=args.sdf_loss,
        replica=args.replica,
        dtu=args.dtu,
        refined=args.refined,
        twodgs=args.twodgs,
        alpha_threshold=args.alpha_threshold,
        step=args.step,
    )

    #mesh_extract.extract_rasterization()
    mesh_extract.extract_marching_cubes()
    #mesh_extract.compute_chamfer_distance()
#     mesh_extract.extract_sdf_plane(
#         origin=(0, 0, 3.5),
#         normal=(0.009955, -0.3814, -0.91903),
#         width=5.0,
#         height=5.0,
#         n=1000
# )