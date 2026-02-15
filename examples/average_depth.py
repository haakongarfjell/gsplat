import os
import json

result_dir = "/home/admin/haakon/gsplat/examples/results"

scan_list = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118, 122]

depth_sdf_total = 0
depth_sdf_rast_total = 0
depth_2dgs_total = 0
depth_3dgs_total = 0

psnr_sdf_total = 0
psnr_2dgs_total = 0
psnr_3dgs_total = 0

ssim_sdf_total = 0
ssim_2dgs_total = 0
ssim_3dgs_total = 0

lpips_sdf_total = 0
lpips_2dgs_total = 0
lpips_3dgs_total = 0



for scan in scan_list:
    sdf_path = os.path.join(result_dir, f"dtu_sdf_scan{scan}", "depth_loss.json")
    sdf_rast_path = os.path.join(result_dir, f"dtu_sdf_scan{scan}", "depth_loss_rasterize.json") 
    dgs_path = os.path.join(result_dir, f"dtu_2dgs_{scan}", "depth_loss.json")
    dgs3_path = os.path.join(f"/media/admin/X9 Pro/results_3dgs/dtu_3dgs_scan{scan}", "depth_loss.json")

    sdf_rgb_path = os.path.join(result_dir, f"dtu_sdf_scan{scan}/stats", "val_step29999.json")
    dgs_rgb_path = os.path.join(result_dir, f"dtu_2dgs_{scan}/stats", "val_step29999.json")
    dgs3_rgb_path = os.path.join(f"/media/admin/X9 Pro/results_3dgs/dtu_3dgs_scan{scan}/stats", "val_step29999.json")
    with open(sdf_path, "r") as f:
        data = json.load(f)
        depth_sdf = data["average_depth_loss"]
        depth_sdf_total += depth_sdf #* 200

    with open(sdf_rast_path, "r") as f:
        data = json.load(f)
        depth_sdf = data["average_depth_loss"]
        depth_sdf_rast_total += depth_sdf #* 200

    with open(sdf_rgb_path, "r") as f:
        data = json.load(f)
        psnr_sdf = data["psnr"]
        psnr_sdf_total += psnr_sdf
        ssim_sdf = data["ssim"]
        ssim_sdf_total += ssim_sdf
        lpips_sdf = data["lpips"]
        lpips_sdf_total += lpips_sdf


    with open(dgs_path, "r") as f:
        data = json.load(f)
        depth_2dgs = data["average_depth_loss"]
        depth_2dgs_total += depth_2dgs #* 200

    with open(dgs_rgb_path, "r") as f:
        data = json.load(f)
        psnr_2dgs = data["psnr"]
        psnr_2dgs_total += psnr_2dgs
        ssim_2dgs = data["ssim"]
        ssim_2dgs_total += ssim_2dgs
        lpips_2dgs = data["lpips"]
        lpips_2dgs_total += lpips_2dgs
        
    with open(dgs3_path, "r") as f:
        data = json.load(f)
        depth_3dgs = data["average_depth_loss"]
        depth_3dgs_total += depth_3dgs

    with open(dgs3_rgb_path, "r") as f:
        data = json.load(f)
        psnr_3dgs = data["psnr"]
        psnr_3dgs_total += psnr_3dgs
        ssim_3dgs = data["ssim"]
        ssim_3dgs_total += ssim_3dgs
        lpips_3dgs = data["lpips"]
        lpips_3dgs_total += lpips_3dgs

print(f"Average PSNR (3DGS): {psnr_3dgs_total / len(scan_list)}")
print(f"Average SSIM (3DGS): {ssim_3dgs_total / len(scan_list)}")
print(f"Average LPIPS (3DGS): {lpips_3dgs_total / len(scan_list)}")
print()

print(f"Average PSNR (SDF): {psnr_sdf_total / len(scan_list)}")
print(f"Average SSIM (SDF): {ssim_sdf_total / len(scan_list)}")
print(f"Average LPIPS (SDF): {lpips_sdf_total / len(scan_list)}")
print()
         
print(f"Average PSNR (2DGS): {psnr_2dgs_total / len(scan_list)}")
print(f"Average SSIM (2DGS): {ssim_2dgs_total / len(scan_list)}")
print(f"Average LPIPS (2DGS): {lpips_2dgs_total / len(scan_list)}")
print()

print(f"Average depth loss (SDF): {depth_sdf_total / len(scan_list)}")
print(f"Average depth loss (SDF Rast): {depth_sdf_rast_total / len(scan_list)}")
print(f"Average depth loss (2DGS): {depth_2dgs_total / len(scan_list)}")
print(f"Average depth loss (3DGS): {depth_3dgs_total / len(scan_list)}")
