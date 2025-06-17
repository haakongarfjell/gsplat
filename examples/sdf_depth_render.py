import numpy as np
import torch
import os
import math
from typing import Dict, Any
import imageio
import cv2


from datasets.dtu_parser import Dataset, Parser, read_pfm
from gsplat.cuda._torch_impl import (
    gaussian_to_ellipse,
    generate_rays,
    calculate_depth,
    generate_depth_image,
    sphere_trace,
)

import torch.nn.functional as F
import re



class SDFDepthRender:

    def __init__(self):
        self.data_dir = "data/DTU/scan24"
        self.depths_gt_dir = os.path.join(self.data_dir, "Depths_raw")

        self.result_dir = "results/dtu_sdf_scan24"
        self.ckpt = "ckpts/ckpt_29999_rank0.pt"
        self.step = 29999

        self.sdf_loss = True
        self.parser = Parser(
            self.data_dir,
            test_every=8,
        )
        self.renderset = Dataset(
            self.parser,
            split="test",
            sdf_loss=self.sdf_loss,
        )

        self.device = "cuda"

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

    def run(self):

        self.load_splat()
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

        print(f"Using {means.shape[0]} Gaussians for sdf computations")
        
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
            print(f"Image ID: {image_id}")
            depth_gt_path = os.path.join(self.depths_gt_dir, f"depth_map_{depth_gt_name}.pfm")
            depth_gt, scale = read_pfm(depth_gt_path)
            depth_gt = cv2.resize(depth_gt, None, fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_NEAREST)
            depth_gt = depth_gt[44:556, 80:720]
            depth_gt = torch.from_numpy(depth_gt).to(self.device)
            depth_gt[~masks] = 0.0

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

            depth = calculate_depth(K, total_distance, width, height) 
            depth[~masks] = 0.0

            threshold = total_distance >= 9.5
            depth[threshold] = 0.0

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

        # write average depth loss to a json file
        with open(os.path.join(self.result_dir, "depth_loss.json"), "w") as f:
            f.write(f'{{"average_depth_loss": {depthloss_total / len(renderloader)}}}\n')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DTU/scan24")
    parser.add_argument("--depths_gt_dir", type=str, default="data/DTU/scan24/Depths_raw")
    parser.add_argument("--result_dir", type=str, default="results/dtu_sdf_scan24")
    parser.add_argument("--ckpt", type=str, default="ckpts/ckpt_29999_rank0.pt")
    parser.add_argument("--step", type=int, default=29999)
    parser.add_argument("--sdf_loss", type=bool, default=True)
    parser.add_argument("--sdf_loss", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    sdf_depth_render = SDFDepthRender(
        data_dir=args.data_dir,
        depths_gt_dir=args.depths_gt_dir,
        result_dir=args.result_dir,
        ckpt=args.ckpt,
        step=args.step,
        sdf_loss=args.sdf_loss,
        device=args.device
    )


    sdf_depth_render.run()

