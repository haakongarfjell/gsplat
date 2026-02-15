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
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

import OpenEXR, Imath, numpy as np

def load_img(img_num):
    img_path = f"/home/admin/haakon/gsplat/examples/data/monkey_depth/exr_files/{img_num}.exr"
    exr_file = OpenEXR.InputFile(img_path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    r_str = exr_file.channel("Composite.Combined.R", pt)
    g_str = exr_file.channel("Composite.Combined.G", pt)
    b_str = exr_file.channel("Composite.Combined.B", pt)
    z_str = exr_file.channel("ViewLayer.Depth.Z", pt)

    r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
    g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
    b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
    z = np.frombuffer(z_str, dtype=np.float32).reshape(height, width)

    rgb = np.stack([r, g, b], axis=-1)
    
    return rgb, z

def linear_to_srgb(image, gamma=2.2):
    image = np.clip(image, 0, None)
    return np.power(image, 1.0/gamma)

def pixel_to_camera(u, v, z, K):
    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    X_cam = (u - c_x) * z / f_x
    Y_cam = (v - c_y) * z / f_y
    Z_cam = z

    points_camera = np.stack([X_cam, Y_cam, Z_cam], axis=-1)

    return points_camera

def cam_to_world_points(points_camera, c2w, H, W):
    ones = np.ones((H, W, 1), dtype=points_camera.dtype)
    points_camera_hom = np.concatenate([points_camera, ones], axis=-1)
    points_camera_hom = points_camera_hom.reshape(-1, 4)
    points_world_hom = (c2w @ points_camera_hom.T).T  # shape (H*W, 4)

    points_world = points_world_hom[:, :3] / points_world_hom[:, 3:]
    points_world = points_world.reshape(H, W, 3)

    return points_world

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
        
        image_dir = os.path.join(data_dir, "exr_files")
        assert os.path.exists(image_dir
        ), f"Image directory {image_dir} does not exist."
        
        cam_path = os.path.join(data_dir, "cameras.txt")

        K = np.array([
            [1111.111111,      0, 400.0],
            [      0,   1111.111111, 400.0],
            [      0,          0,     1.0]
        ])

        # K = np.array([
        #     [1111.111111,      0, 400.0],
        #     [      0,   1666.666667, 400.0],
        #     [      0,          0,     1.0]
        # ])
        K[:2, :] /= factor
        
        cams = np.loadtxt(cam_path, delimiter=",", comments="#")
        positions = cams[:, 1:4]
        euler_angles = cams[:, 4:7]
        rotations = Rotation.from_euler("xyz", euler_angles, degrees=False).as_matrix()

        N = positions.shape[0]
        camtoworlds_m= np.zeros((N, 4, 4))
        camtoworlds_m[:, :3, :3] = rotations      
        camtoworlds_m[:, :3, 3] = positions      
        camtoworlds_m[:, 3, 3] = 1.0 

        T = np.diag([1, -1, -1, 1])

        camtoworlds = camtoworlds_m @ T

        H, W = (800, 800)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        # Extract extrinsic matrices in world-to-camera format.

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
        for i in range(1, N):
            if i < 9:
                img_id = f"000{i+1}"
            elif i < 99:
                img_id = f"00{i+1}"
            elif i < 999:
                img_id = f"0{i+1}"
            else:
                img_id = f"{i+1}"
            
            image_names.append(img_id)

            w2c = np.linalg.inv(camtoworlds[i])
            w2c_mats.append(w2c)
            camera_id = int(cams[i, 0])
            camera_ids.append(camera_id)

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            Ks_dict[camera_id] = K

            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (W // factor, H // factor)
            mask_dict[camera_id] = None

            rgb_m, z = load_img(img_id)
            rgb = linear_to_srgb(rgb_m)
            points_camera = pixel_to_camera(u, v, z, K)

            points_world = cam_to_world_points(points_camera, camtoworlds[i], H, W)
            valid_indices = (z > 0) & (z < 5000)

            points_world = points_world[valid_indices]
            rgb = rgb[valid_indices]

            points_per_image[img_id] = points_world

            points3D.append(points_world)
            rgbs.append(rgb)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            pcd.orient_normals_towards_camera_location(camtoworlds[i][:3, 3])
            normals_world = np.asarray(pcd.normals)
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
        colmap_image_dir = os.path.join(data_dir, "exr_files")
        image_dir = os.path.join(data_dir, "exr_files" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        image_paths = [os.path.join(data_dir, "exr_files", f"{img_id}.exr") for img_id in image_names]

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

            T2 = align_principal_axes(points)
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
        image, z = load_img(self.parser.image_names[index])
        image = linear_to_srgb(image)

        image = (image * 255).astype(np.uint8)
        
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

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
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            points_world = points_world[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
            data["points_world"] = torch.from_numpy(points_world).float()

        if self.sdf_loss:
            image_name = self.parser.image_names[index]
            points_world = self.parser.points_per_image[image_name]
            data["points_world"] = torch.from_numpy(points_world).float()
            

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
