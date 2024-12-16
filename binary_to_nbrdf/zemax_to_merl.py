import numpy as np
import math
import struct
import scipy.interpolate as interpolate
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import os

# Calculate the index for theta_half.
def theta_half_index(theta_half, BRDF_SAMPLING_RES_THETA_H=90):
    if theta_half <= 0.0:
        return 0
    theta_half_deg = ((theta_half / (math.pi/2.0))*BRDF_SAMPLING_RES_THETA_H)
    temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H
    temp = math.sqrt(temp)
    ret_val = int(temp)
    if ret_val < 0:
        ret_val = 0
    if ret_val >= BRDF_SAMPLING_RES_THETA_H:
        ret_val = BRDF_SAMPLING_RES_THETA_H-1
    return ret_val

# Calculate the index for theta_diff.
def theta_diff_index(theta_diff, BRDF_SAMPLING_RES_THETA_D=90):
    tmp = int(theta_diff / (math.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D)
    if tmp < 0:
        return 0
    elif tmp < BRDF_SAMPLING_RES_THETA_D - 1:
        return tmp
    else:
        return BRDF_SAMPLING_RES_THETA_D - 1
    
# Calculate the index for phi_diff.
def phi_diff_index(phi_diff, BRDF_SAMPLING_RES_PHI_D=360):
    if phi_diff < 0.0:
        phi_diff += math.pi  # Use math.pi for consistency
    tmp = int(phi_diff / math.pi * BRDF_SAMPLING_RES_PHI_D / 2)
    if tmp < 0:
        return 0
    elif tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1:
        return tmp
    else:
        return BRDF_SAMPLING_RES_PHI_D / 2 - 1
    
# Rotate a vector around an axis.
def rotate_vector(vector, axis, angle):
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    out = vector * cos_ang
    temp = np.dot(axis, vector) * (1.0 - cos_ang)
    out += axis * temp
    cross = np.cross(axis, vector)
    out += cross * sin_ang
    return out

# Convert standard coordinates to half-diff coordinates.
def std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out):
    # Compute in vector
    in_vec_z = np.cos(theta_in)
    proj_in_vec = np.sin(theta_in)
    in_vec_x = proj_in_vec * np.cos(fi_in)
    in_vec_y = proj_in_vec * np.sin(fi_in)
    in_vec = np.array([in_vec_x, in_vec_y, in_vec_z])
    in_vec = in_vec / np.linalg.norm(in_vec)  # Normalize

    # Compute out vector
    out_vec_z = np.cos(theta_out)
    proj_out_vec = np.sin(theta_out)
    out_vec_x = proj_out_vec * np.cos(fi_out)
    out_vec_y = proj_out_vec * np.sin(fi_out)
    out_vec = np.array([out_vec_x, out_vec_y, out_vec_z])
    out_vec = out_vec / np.linalg.norm(out_vec)  # Normalize

    # Compute halfway vector
    half_vec = (in_vec + out_vec) / 2.0
    half_vec = half_vec / np.linalg.norm(half_vec)  # Normalize

    # Compute theta_half, fi_half
    theta_half = np.arccos(half_vec[2])
    fi_half = np.arctan2(half_vec[1], half_vec[0])

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    # Compute diff vector
    temp = rotate_vector(in_vec, normal, -fi_half)
    diff_vec = rotate_vector(temp, bi_normal, -theta_half)

    # Compute  theta_diff, fi_diff
    theta_diff = np.arccos(diff_vec[2])
    fi_diff = np.arctan2(diff_vec[1], diff_vec[0])

    return theta_half, theta_diff, fi_diff

# Convert Zemax data to MERL format.
def convert_to_merl(interpolated_data, resolution=(90, 90, 180)):
    theta_h, theta_d, phi_d = resolution
    binary_data = struct.pack('3i', theta_h, theta_d, phi_d)  # Write the resolution header
    
    # Write out each entry in the form r, g, b
    for i in range(theta_h):
        print(i)
        for j in range(theta_d):
            for k in range(phi_d):
                r, g, b = interpolated_data[i][j][k]
                binary_data += struct.pack('3d', r, g, b)  # Each value as double (float64)
    
    return binary_data

# Write interpolated Zemax data as a MERL binary file.
def write_merl_file(output_path, interpolated_data, resolution=(90, 90, 180)):
    binary_data = convert_to_merl(interpolated_data, resolution)
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    print(f"MERL file successfully written to {output_path}")

# Parse a Zemax .brdf file and extract the relevant data.
def parse_zemax_to_merl_format(zemax_file):
    with open(zemax_file, 'r') as f:
        lines = f.readlines()

    theta_i = []
    phi_i = []
    theta_o = []
    brdf_data = []

    for i, line in enumerate(lines):
        if line.startswith("AngleOfIncidence"):
            theta_i = np.array([float(x) for x in lines[i + 1].split()])
        elif line.startswith("ScatterAzimuth"):
            phi_i = np.array([float(x) for x in lines[i + 1].split()])
        elif line.startswith("ScatterRadial"):
            theta_o = np.array([float(x) for x in lines[i + 1].split()])
        elif line.startswith("DataBegin"):
            data_block = []
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("DataEnd"):
                    break
                if lines[j].startswith("TIS"):
                    continue
                data_block.append([float(x) for x in lines[j].split()])
            brdf_data.extend(data_block)  # Extend the list instead of appending

    # Reshape the BRDF data into a 4D array (theta_o, theta_i, phi_i, XYZ)
    num_theta_i = len(theta_i)
    num_phi_i = len(phi_i)
    num_theta_o = len(theta_o)
    brdf_data = np.array(brdf_data)

    return theta_i, phi_i, theta_o, brdf_data


if __name__ == "__main__":
    folder_path = '/Users/abraham/Desktop/ki'
    brdf_files = [f for f in os.listdir(folder_path) if f.endswith('.brdf')]

    for brdf_file in brdf_files:
        file_path = os.path.join(folder_path, brdf_file)
        print(f"Processing {file_path}...")

        theta_i, phi_i, theta_o, brdf_data = parse_zemax_to_merl_format(file_path)

        theta_i = theta_i[:-1]  # Remove last element if needed
        merl = np.full((90, 90, 180, 3), fill_value=np.nan)  # Change "not set" value to NaN for better handling

        for i in range(len(theta_i)):
            for j in range(len(phi_i)):
                for k in range(len(theta_o)):
                    theta_i_r = np.radians(theta_i[i])
                    phi_i_r = np.radians(phi_i[j])
                    theta_o_r = np.radians(theta_o[k])
                    phi_o_r = 0  # Assuming phi_o is always 0

                    theta_h, theta_d, phi_d = std_coords_to_half_diff_coords(theta_i_r, phi_i_r, theta_o_r, phi_o_r)
                    
                    reflectanceX = brdf_data[0 * len(phi_i) * len(theta_i) + i * len(phi_i) + j][k]
                    reflectanceY = brdf_data[1 * len(phi_i) * len(theta_i) + i * len(phi_i) + j][k]
                    reflectanceZ = brdf_data[2 * len(phi_i) * len(theta_i) + i * len(phi_i) + j][k]

                    merl[int(theta_half_index(theta_h))][int(theta_diff_index(theta_d))][int(phi_diff_index(phi_d))][0] = reflectanceX
                    merl[int(theta_half_index(theta_h))][int(theta_diff_index(theta_d))][int(phi_diff_index(phi_d))][1] = reflectanceY
                    merl[int(theta_half_index(theta_h))][int(theta_diff_index(theta_d))][int(phi_diff_index(phi_d))][2] = reflectanceZ

        known_points = []
        known_values = []
        grid_points = []

        for i in range(90):
            for j in range(90):
                for k in range(180):
                    grid_points.append([i, j, k])
                    if not np.isnan(merl[i][j][k][0]): 
                        known_points.append([i, j, k])
                        known_values.append(merl[i][j][k])

        # **Step 1: Linear Interpolation**
        print("Performing initial linear interpolation...")
        interpolated_values = interpolate.griddata(
            np.array(known_points), 
            np.array(known_values), 
            np.array(grid_points), 
            method='linear'
        )
        interpolated_values = interpolated_values.reshape(90, 90, 180, 3)
        print("...done with linear interpolation")

        # **Step 2: Update known_points and known_values to exclude NaN**
        new_known_points = []
        new_known_values = []
        for i in range(90):
            for j in range(90):
                for k in range(180):
                    if not np.isnan(interpolated_values[i, j, k, 0]):  # Check if the first channel (R) is NaN
                        new_known_points.append([i, j, k])
                        new_known_values.append(interpolated_values[i, j, k])
        
        # **Step 3: Nearest-Neighbor Interpolation to Fill NaNs**
        print("Filling NaNs with nearest-neighbor interpolation...")
        final_interpolated_values = interpolate.griddata(
            np.array(new_known_points), 
            np.array(new_known_values), 
            np.array(grid_points), 
            method='nearest'
        )
        final_interpolated_values = final_interpolated_values.reshape(90, 90, 180, 3)
        print("...done with nearest-neighbor interpolation")

        # Plot the 2D slices
        phi_d_slices = [0, 45, 90]  # Indices for phi_d slices
        for i, phi_d in enumerate(phi_d_slices):
            slice_rgb = final_interpolated_values[:, :, phi_d, :]  # Extract RGB slice
            
            plt.figure(figsize=(12, 4))
            
            # Plot combined RGB image
            plt.subplot(1, 4, 1)
            plt.imshow(slice_rgb)
            plt.title(f'{brdf_file} - phi_d = {phi_d} (RGB)')
            plt.axis('off')

            # Plot individual R, G, B channels
            for j, color in enumerate(['R', 'G', 'B']):
                plt.subplot(1, 4, j + 2)
                plt.imshow(slice_rgb[:, :, j], cmap='inferno')
                plt.title(f'{color} channel')
                plt.axis('off')

            plt.suptitle(f'2D BRDF Slice for {brdf_file} at phi_d = {phi_d}°', fontsize=16)
            
            # Create plots folder if it doesn't exist
            output_folder = os.path.join(folder_path, 'plots')
            os.makedirs(output_folder, exist_ok=True)

            # Save the plot with a file-specific name
            plot_filename = os.path.join(output_folder, f'{brdf_file}_phi_{phi_d}.png')
            plt.savefig(plot_filename)
            plt.close()  # Close plot to free memory

        # Create the MERL binary file
        binary_filename = os.path.join(folder_path, f'{brdf_file}.binary')
        write_merl_file(binary_filename, final_interpolated_values)