import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never
from scipy.spatial.transform import Rotation

import open3d as o3d
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

import re

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape) * 1/200
    data = np.flipud(data)
    file.close()
    return data, scale

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

def unproject_depths(depth, image, mask, w2c, K, near_fars):
    
    depth_h, depth_w = depth.shape  # (128, 160)
    # Image resolution:
    image_h, image_w = image.shape[0], image.shape[1]  # (512, 640)

    # Compute scaling factors:
    scale_x = image_w / depth_w  # 640 / 160 = 4
    scale_y = image_h / depth_h  # 512 / 128 = 4

    v_depth, u_depth = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
    u_img = u_depth * scale_x
    v_img = v_depth * scale_y


    u_img_flat = u_img.flatten()
    v_img_flat = v_img.flatten()
    depth_flat = depth.flatten() 

    far = near_fars[0]
    near = near_fars[1]

    valid = (depth_flat >= near) & (depth_flat <= far)

    # Apply filtering to the pixel coordinates and depth values
    u_img_flat = u_img_flat[valid]
    v_img_flat = v_img_flat[valid]
    depth_flat = depth_flat[valid]

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    X = (u_img_flat - cx) * depth_flat / fx
    Y = (v_img_flat - cy) * depth_flat / fy
    Z = depth_flat

    points_camera = np.stack([X, Y, Z], axis=-1)

    w2c_inv = np.linalg.inv(w2c)
    points_cam_hom = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=-1)
    points_world_hom = (w2c_inv @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3] / points_world_hom[:, 3:4]

    u_indices = np.round(u_img_flat).astype(np.int32)
    v_indices = np.round(v_img_flat).astype(np.int32)
    
    # It is important to check that the indices are in range.
    # (In theory they are, if the downsampling/cropping is aligned.)
    colors = image[v_indices, u_indices, :]  # image is indexed as [row, col]
    colors = image[  v_indices, u_indices, : ]   # (N,3) or (N,4)

    return points_world, colors

def voxel_downsample_unique(points, rgbs, normals, num_bins=50):

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    edges = [np.linspace(min_coords[d], max_coords[d], int(num_bins)+1) for d in range(3)]
    
    bin_x = np.digitize(points[:, 0], edges[0]) - 1
    bin_y = np.digitize(points[:, 1], edges[1]) - 1
    bin_z = np.digitize(points[:, 2], edges[2]) - 1
    
    bin_x = np.clip(bin_x, 0, num_bins-1)
    bin_y = np.clip(bin_y, 0, num_bins-1)
    bin_z = np.clip(bin_z, 0, num_bins-1)
    
    cell_indices = np.stack([bin_x, bin_y, bin_z], axis=1)
    
    cell_dict = {}
    for idx, cell in enumerate(map(tuple, cell_indices)):
        if cell not in cell_dict:
            cell_dict[cell] = []
        cell_dict[cell].append(idx)
    
    selected_idx = []
    for cell, indices in cell_dict.items():

        selected_idx.append(indices[0])
    
    selected_idx = np.array(selected_idx)
    
    return points[selected_idx], rgbs[selected_idx], normals[selected_idx]

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:


    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        scan_name   = os.path.basename(os.path.normpath(data_dir))  
        # scan_name == "scan24"

        depths_dir  = os.path.join(
            data_dir,
            "Depths",
            f"{scan_name}_train"
        )

        #depths_dir = os.path.join(data_dir, f"Depths/{data_dir[:-1]}_train")
        image_dir = os.path.join(data_dir, "Rectified")
        cam_dir = os.path.join(data_dir, "Cameras/train")
        mask_dir = os.path.join(data_dir, "mask")



        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        image_names = []

        points3D = []
        rgbs = []
        normals3D = []
        points_per_image = dict()
        normals_per_image = dict()

        N = 49
        for image_number in range(0, N):

            depth_name = f"{image_number:04d}"
            image_name = f"{(image_number+1):03d}"
            img_id = image_name
            mask_name = f"{(image_number):03d}"
            cam_name = f"{image_number:08}"

            depth_path = os.path.join(depths_dir, f"depth_map_{depth_name}.pfm")
            image_path = os.path.join(image_dir, f"rect_{image_name}_3_r5000.png")
            mask_path  = os.path.join(mask_dir, f"{mask_name}.png")
            cam_path = os.path.join(cam_dir, f"{cam_name}_cam.txt")
            
            image_names.append(image_name)

            depth, scale = read_pfm(depth_path)
            image = imageio.imread(image_path)
            w2c, K, near_fars = read_cam(cam_path)
            c2w = np.linalg.inv(w2c)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
            mask = mask[44:556, 80:720]

            H, W = image.shape[0], image.shape[1]
            w2c_mats.append(w2c)
            camera_id = image_number
            camera_ids.append(camera_id)

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            Ks_dict[camera_id] = K

            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (W // factor, H // factor)
            mask_dict[camera_id] = None

            points_world, rgb = unproject_depths(depth, image, mask, w2c, K, near_fars)
            points_per_image[img_id] = points_world

            points3D.append(points_world)
            rgbs.append(rgb)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            pcd.orient_normals_towards_camera_location(c2w[:3, 3])
            normals_world = np.asarray(pcd.normals)
            normals_per_image[img_id] = normals_world

            normals3D.append(normals_world)


        print(
            f"[Parser] {N} images, taken by {len(set(camera_ids))} cameras."
        )

        points3D = np.concatenate(points3D, axis=0).reshape(-1, 3)
        rgbs = np.concatenate(rgbs, axis=0).reshape(-1, 3)
        normals_all = np.concatenate(normals3D, axis=0).reshape(-1, 3)

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "Rectified")
        image_dir = os.path.join(data_dir, "Rectified" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        image_paths = [os.path.join(data_dir, "Rectified", f"{img_id}.png") for img_id in image_names]

        points, points_rgb, normals = voxel_downsample_unique(points3D, rgbs, normals_all)

        points_err = np.zeros(points.shape[0], dtype=np.float32)

        point_indices = dict()
        
        for i, img_id in enumerate(image_names):
            image_name = img_id
            point_indices.setdefault(image_name, [])

        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.normals = normals  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)
        self.points_per_image = points_per_image  # Dict[str, np.ndarray], image_name -> [M, 3]
        self.normals_per_image = normals_per_image
        self.points_all = points3D

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_height, actual_width = (H, W)
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = (H, W)
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        sdf_loss: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.sdf_loss = sdf_loss
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        img_name_full = f"{self.parser.data_dir}/Rectified/rect_{self.parser.image_names[index]}_3_r5000.png"

        image = imageio.imread(img_name_full)[..., :3]        
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]


        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        data["points_all"] = torch.from_numpy(self.parser.points_all).float()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            #point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points_per_image[image_name]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
                      # filter out points outside the image
   
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
            data["points_world"] = torch.from_numpy(points_world).float()

        if self.sdf_loss:
            image_name = self.parser.image_names[index]
            points_world = self.parser.points_per_image[image_name]
            normals_world = self.parser.normals_per_image[image_name]
            data["points_world"] = torch.from_numpy(points_world).float()
            data["normals_world"] = torch.from_numpy(normals_world).float()
            

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True, sdf_loss=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()