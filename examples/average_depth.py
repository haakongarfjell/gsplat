import os
import json

result_dir = "/home/admin/haakon/gsplat/examples/results"

scan_list = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118, 122]

depth_sdf_total = 0
depth_2dgs_total = 0

for scan in scan_list:
    sdf_path = os.path.join(result_dir, f"dtu_sdf_scan{scan}", "depth_loss.json")
    dgs_path = os.path.join(result_dir, f"dtu_2dgs_{scan}", "depth_loss.json")
    with open(sdf_path, "r") as f:
        data = json.load(f)
        depth_sdf = data["average_depth_loss"]
        depth_sdf_total += depth_sdf


    with open(dgs_path, "r") as f:
        data = json.load(f)
        depth_2dgs = data["average_depth_loss"]
        depth_2dgs_total += depth_2dgs

print(f"Average depth loss (SDF): {depth_sdf_total / len(scan_list)}")
print(f"Average depth loss (2DGS): {depth_2dgs_total / len(scan_list)}")