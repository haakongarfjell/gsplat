import os
import json

result_dir = "/home/admin/haakon/gsplat/examples/results"

scan_list = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118, 122]

total_6999 = 0
total_29999 = 0

total_s2d_6999 = 0
total_s2d_29999 = 0

total_d2s_6999 = 0
total_d2s_29999 = 0

for scan in scan_list:
    scan_dir = os.path.join(result_dir, f"dtu_sdf_scan{scan}")

    vis_6999 = os.path.join(scan_dir, "vis_6999")
    vis_29999 = os.path.join(scan_dir, "vis_29999")

    with open(os.path.join(vis_6999, "results.json"), "r") as f:
        data_6999 = json.load(f)

        overall_6999 = data_6999["overall"]
        mean_s2d_6999 = data_6999["mean_s2d"]
        mean_d2s_6999 = data_6999["mean_d2s"]
        total_6999 += overall_6999
        total_s2d_6999 += mean_s2d_6999
        total_d2s_6999 += mean_d2s_6999

    with open(os.path.join(vis_29999, "results.json"), "r") as f:
        data_29999 = json.load(f)

        overall_29999 = data_29999["overall"]
        mean_s2d_29999 = data_29999["mean_s2d"]
        mean_d2s_29999 = data_29999["mean_d2s"]
        total_29999 += overall_29999
        total_s2d_29999 += mean_s2d_29999
        total_d2s_29999 += mean_d2s_29999

print(f"Average 6999: {total_6999 / len(scan_list)}")
print(f"Average 29999: {total_29999 / len(scan_list)}")

print(f"Average s2d 6999: {total_s2d_6999 / len(scan_list)}")
print(f"Average s2d 29999: {total_s2d_29999 / len(scan_list)}")

print(f"Average d2s 6999: {total_d2s_6999 / len(scan_list)}")
print(f"Average d2s 29999: {total_d2s_29999 / len(scan_list)}")


