import numpy as np
import torch
import os
import math
import imageio
import cv2
from typing import Dict, List, Literal, Optional, Tuple, Any
from torch import Tensor


from datasets.dtu_parser import Dataset, Parser, read_pfm
from gsplat.cuda._torch_impl import (
    gaussian_to_ellipse,
    generate_rays,
    calculate_depth,
    generate_depth_image,
    sphere_trace,
)

from gsplat.rendering import rasterization_2dgs

import torch.nn.functional as F
import re

import open3d as o3d
import scipy.io as sio

class SDFDepthRender:

    def __init__(self,
        data_dir: str,
        depths_gt_dir: str,
        result_dir: str,
        ckpt: str,
        step: int,
        sdf_loss: bool = False,
        load_depths: bool = False,
        device: str = "cuda",
    ):
        self.data_dir = data_dir
        self.depths_gt_dir = depths_gt_dir
        self.result_dir = result_dir
        self.ckpt = ckpt
        self.step = step
        self.sdf_loss = sdf_loss
        self.load_depths = load_depths
        self.device = device

        self.parser = Parser(
            self.data_dir,
            test_every=1,
        )
        self.renderset = Dataset(
            self.parser,
            split="test",
            load_depths=self.load_depths,
            sdf_loss=self.sdf_loss,
        )

        self.patch = 0.0
        # pcd = o3d.io.read_point_cloud(self.stl_path)
        # pts = np.asarray(pcd.points) / 200.0
        # print(np.min(pts, axis=0), np.max(pts, axis=0))
        # self.points_gt_cloud = pts

        scan = int(os.path.basename(self.data_dir).lstrip("scan"))
        obs_mat = sio.loadmat(f"data/DTU/ObsMask/ObsMask{scan}_10.mat")
        P_mat   = sio.loadmat(f"data/DTU/ObsMask/Plane{scan}.mat")

        # original units are mm → convert to metres
        BB = obs_mat["BB"].astype(np.float32)   # shape (2,3)
        Res   = obs_mat["Res"].astype(np.float32).ravel()  # (3,)
        P     = P_mat["P"].astype(np.float32).ravel()      # (4,)

        scale = 1.0 / 200.0
        BB *= scale       # now in metres
        Res *= scale      # now in metres per voxel

        # plane equation P·[x,y,z,1] > 0 also needs to be re-scaled.
        # Easiest: divide the entire 4-vector by 200 (any nonzero scalar preserves the half-space):
        P   *= scale

        self.ObsMask = obs_mat["ObsMask"].astype(bool)
        self.BB_min, self.BB_max = BB[0], BB[1]
        self.Res  = Res
        self.plane = P

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
    
    def stl_render(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
    ):
        c2w = camtoworlds.squeeze(0).cpu().numpy()           # (4,4)
        w2c = np.linalg.inv(c2w)
        pts_cam = (w2c[:3,:3] @ self.points_gt_cloud.T + w2c[:3,3:4]).T  # (M,3)
        valid_pts = pts_cam[:,2] > 0
        pts_cam = pts_cam[valid_pts]

        K_np = Ks.squeeze(0).cpu().numpy()                   # (3,3)
        proj = (K_np @ pts_cam.T).T                          # (M, 3)
        uv  = proj[:, :2] / proj[:, 2:3]                     # (M,2)
        u   = np.round(uv[:,0]).astype(int)
        v   = np.round(uv[:,1]).astype(int)
        z   = pts_cam[:,2]

        depth_gt_np = np.zeros((height, width), dtype=np.float32)
        for ui, vi, zi in zip(u, v, z):
            if 0 <= ui < width and 0 <= vi < height:
                prev = depth_gt_np[vi, ui]
                if prev == 0 or zi < prev:
                    depth_gt_np[vi, ui] = zi
        
        return torch.from_numpy(depth_gt_np).to(self.device)
    
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

    def rasterize_splats(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
        )

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    def run(self):

        self.load_splat()

        if self.sdf_loss:
            self.load_gaussian_ids()
            means = self.means[self.gaussian_ids]
            quats = self.quats[self.gaussian_ids]
            scales = self.scales[self.gaussian_ids]
            opacities = self.opacities[self.gaussian_ids]
            sdf_coeffs = self.sdf_coeffs[self.gaussian_ids]
            colors = self.colors[self.gaussian_ids]

            opacity_threshold = 0.1
            mask = opacities > opacity_threshold
            means = means[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            sdf_coeffs = sdf_coeffs[mask]
            colors = colors[mask]

            r_a, r_b, axes_a, axes_b, _ = gaussian_to_ellipse(
                means, 
                quats, 
                scales, 
                opacities
            )
        
        
        renderloader = torch.utils.data.DataLoader(
            self.renderset, batch_size=1, shuffle=False, num_workers=1
        )

        video_path = os.path.join(self.result_dir, "depth.mp4")
        writer = imageio.get_writer(video_path, fps=30)

        depthloss_total = 0.0
        for i, data in enumerate(renderloader):
            print(f"Processing {i+1}/{len(renderloader)}")
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data["mask"].to(self.device).squeeze(0)
            height, width = pixels.shape[1:3]

            image_id = data["image_id"].item()
            depth_gt_name = f"{i:04d}"
            depth_gt_path = os.path.join(self.depths_gt_dir, f"depth_map_{depth_gt_name}.pfm")
            depth_gt, scale = read_pfm(depth_gt_path)
            depth_gt = cv2.resize(depth_gt, None, fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_NEAREST)
            depth_gt = depth_gt[44:556, 80:720]
            depth_gt = torch.from_numpy(depth_gt).to(self.device)

            #depth_gt = self.stl_render(camtoworlds, Ks, width, height)
            #depth_gt[~masks] = 0.0

            if self.sdf_loss:
                c2w = camtoworlds.squeeze(0)
                K = Ks.squeeze(0)


                origins, directions = generate_rays(c2w, K, width, height)
                total_distance = sphere_trace(
                    means,
                    r_a,
                    r_b,
                    axes_a,
                    axes_b,
                    sdf_coeffs,
                    self.sh_degree,
                    origins,
                    directions,
                )

                depth_pred = calculate_depth(K, total_distance, width, height) 
                depth_pred[~masks] = 0.0

                threshold = total_distance >= 9.5
                depth_pred[threshold] = 0.0
            # render depth here
            else:
                (
                    colors,
                    alphas,
                    normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    _,
                )= self.rasterize_splats(
                    means=self.means,
                    quats=self.quats,
                    scales=self.scales,
                    opacities=self.opacities,
                    colors=self.colors,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=self.sh_degree,
                    # near_plane=0.2,
                    # far_plane=200.0,
                )  # [1, H, W, 4]

                depth_pred = render_median.squeeze(0).squeeze(-1).detach()  # [H, W]
   

            H, W         = depth_gt.shape
            depth_gt_np  = depth_gt.cpu().numpy()
            masks_np     = masks.cpu().numpy()
            K_np         = Ks[0].cpu().numpy()           # (3,3)
            c2w_np       = camtoworlds[0].cpu().numpy()  # (4,4)
            inv_c2w      = np.linalg.inv(c2w_np)

            ys, xs      = np.indices((H, W))
            valid       = (depth_gt_np > 0) & masks_np
            flat_valid  = np.nonzero(valid.ravel())[0]

            z_cam       = depth_gt_np.ravel()[flat_valid]
            x_cam       = (xs.ravel()[flat_valid] - K_np[0,2]) * z_cam / K_np[0,0]
            y_cam       = (ys.ravel()[flat_valid] - K_np[1,2]) * z_cam / K_np[1,1]
            ones        = np.ones_like(z_cam)
            pts_cam_h   = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)  # (M,4)

            pts_w_h     = (inv_c2w @ pts_cam_h.T).T                      # (M,4)
            pts_w       = pts_w_h[:,:3] / pts_w_h[:,3:4]                 # (M,3)

            grid        = np.floor((pts_w - self.BB_min) / self.Res).astype(int)
            grid        = np.minimum(np.maximum(grid, 0), np.array(self.ObsMask.shape)-1)
            #obs_ok      = self.ObsMask[grid[:,0], grid[:,1], grid[:,2]]

            pts_w_h4    = np.concatenate([pts_w, np.ones((pts_w.shape[0],1))], axis=1)
            above       = (pts_w_h4 * self.plane).sum(axis=1) > 0

            keep        = above
            idxs        = flat_valid[keep]
            eval_mask   = np.zeros(H*W, dtype=bool)
            eval_mask[idxs] = True
            eval_mask   = eval_mask.reshape(H, W)

            m           = torch.from_numpy(eval_mask).to(self.device)
            depth_gt    = depth_gt   * m
            depth_pred  = depth_pred * m

            depth = depth_pred

            depth_loss = F.l1_loss(depth, depth_gt)
            depthloss_total += depth_loss.item()
            print(f"Depth Loss: {depth_loss.item()}")


            depth_normalized = torch.zeros_like(depth)
            depth_gt_normalized = torch.zeros_like(depth_gt)

            depth_normalized[masks]     = (depth[masks]     - depth[masks].min()) / (depth[masks].max() - depth[masks].min())
            depth_gt_normalized[masks]  = (depth_gt[masks]  - depth_gt[masks].min()) / (depth_gt[masks].max() - depth_gt[masks].min())
            # depth_normalized    = torch.clamp(depth_normalized,    0.0, 1.0)
            # depth_gt_normalized = torch.clamp(depth_gt_normalized, 0.0, 1.0)

            err       = torch.abs(depth - depth_gt)                                         # [H, W]
            err       = err.cpu().numpy()
            max_err   = np.percentile(err, 99)                                               # or a fixed value
            err_norm  = np.clip(err / max_err, 0.0, 1.0)
            err_uint8 = (err_norm * 255).astype(np.uint8)                                   # [H, W]

            heat_bgr = cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET)                       # [H, W, 3]
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)                             # [H, W, 3]

            depth_pred_uint8 = (depth_normalized.cpu().numpy() * 255).astype(np.uint8)  # [H, W]
            depth_gt_uint8   = (depth_gt_normalized.cpu().numpy() * 255).astype(np.uint8) # [H, W]
            depth_pred_rgb   = np.stack([depth_pred_uint8]*3, axis=-1)                  # [H, W, 3]
            depth_gt_rgb     = np.stack([depth_gt_uint8]*3,   axis=-1)                  # [H, W, 3]

            frame = np.concatenate([depth_pred_rgb,
                                    depth_gt_rgb,
                                    heat_rgb], axis=0)                   
            writer.append_data(frame)
        writer.close()
        print(f"Saved depth video to {video_path}")
        print(f"Average depth loss: {depthloss_total / len(renderloader)}")
        with open(os.path.join(self.result_dir, "depth_loss.json"), "w") as f:
            f.write(f'{{"average_depth_loss": {depthloss_total / len(renderloader)}}}\n')




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DTU/scan24")
    parser.add_argument("--depths_gt_dir", type=str, default="data/DTU/scan24/Depths_raw")
    parser.add_argument("--results_dir", type=str, default="results/dtu_sdf_scan24")
    parser.add_argument("--ckpt", type=str, default="ckpts/ckpt_29999_rank0.pt")
    parser.add_argument("--step", type=int, default=29999)
    parser.add_argument("--sdf_loss", type=bool, default=False)
    parser.add_argument("--load_depths", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    sdf_depth_render = SDFDepthRender(
        data_dir=args.data_dir,
        depths_gt_dir=args.depths_gt_dir,
        result_dir=args.results_dir,
        ckpt=args.ckpt,
        step=args.step,
        sdf_loss=args.sdf_loss,
        device=args.device
    )


    sdf_depth_render.run()

