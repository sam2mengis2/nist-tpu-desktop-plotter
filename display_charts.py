import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QLabel, QVBoxLayout, QWidget, QFrame)
from PyQt6.QtCore import Qt
import pandas as pd
import io
from scipy.interpolate import griddata
import numpy as np
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyhdf.SD import SD, SDC
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from itertools import islice

class NIST_DATA_ANALYZER:
    def __init__(self, hdf_path):
        self.hdf_path = hdf_path
        self.df = None

    def extract_dataframes(self):
        """
        Opens an HDF4 file, extracts all datasets dynamically, 
        and returns a dictionary of Pandas DataFrames.
        """
        # Dictionary to store our final DataFrames
        extracted_dfs = {}

        try:
            # 1. Open the file ONCE at the start
            hdf_file = SD(self.hdf_path, SDC.READ)
            datasets = hdf_file.datasets()

            print(f"Success! Found {len(datasets)} datasets. Processing extraction...\n")
            
            # 2. Loop through every dataset found in the file dynamically
            for name in datasets.keys():
                try:
                    # Select the specific dataset
                    dataset = hdf_file.select(name)
                    
                    # Extract the raw Numpy array
                    raw_data = dataset.get()
                    
                    # Convert to Pandas DataFrame
                    df = pd.DataFrame(raw_data)
                    
                    # Store in our dictionary
                    extracted_dfs[name] = df
                    print(f"✅ Extracted '{name}' -> DataFrame Shape: {df.shape}")
                    
                    # Close the specific dataset connection
                    dataset.endaccess()
                    
                except Exception as e:
                    print(f"❌ Could not extract '{name}'. Error: {e}")
                    
            # 3. Close the master file connection ONCE at the very end
            hdf_file.end()
            return extracted_dfs
            
        except Exception as e:
            print(f"Critical Error opening or processing file: {e}")
            return None
        

    def view_history_at_time(self, pressure_time_series, tap_no, time_a, time_b):
        tap_idx = int(tap_no)
        start_idx = int(time_a)
        end_idx = int(time_b)


        pressure_tap_no = pressure_time_series[tap_idx]

        sliced_pressure = pressure_tap_no.iloc[start_idx: end_idx]

        plt.plot(
            sliced_pressure.index, 
            sliced_pressure.values, 
            color='#1f77b4', 
            linewidth=0.5, 
            label=f'Tap Channel {tap_no}'
        )

        plt.title(f'Pressure Coefficient Time Series - Tap {tap_no}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Timestep (Frames / Samples)', fontsize=10)
        plt.ylabel('Pressure Value ($C_p$)', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')

        plt.tight_layout()

        plt.show()


    def view_full_series_tap(self, pressure_time_series, tap_no):
        tap_idx = int(tap_no)

        pressure_at_tap = pressure_time_series[tap_idx]


        plt.plot(
            pressure_at_tap.index, 
            pressure_at_tap.values, 
            color='#1f77b4', 
            linewidth=0.5, 
            label=f'Tap Channel {tap_no}'
        )

        
        plt.title(f'Pressure Coefficient Time Series - Tap {tap_no}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Timestep (Frames / Samples)', fontsize=10)
        plt.ylabel('Pressure Value ($C_p$)', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')

        plt.tight_layout()

        plt.show()

    def get_wind_frame_plot_3D(self, tap_df, frame_df, corners_df):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(corners_df[0], corners_df[1], corners_df[2], 
            color='darkorange', s=60, label='Corners')
        
        for _, line in frame_df.iterrows():
            start_id = int(line[0]) - 1
            end_id = int(line[1]) - 1


            start_node = corners_df.iloc[start_id]
            end_node = corners_df.iloc[end_id]

        
            xs = [start_node[0], end_node[0]]
            ys = [start_node[1], end_node[1]]
            zs = [start_node[2], end_node[2]]

            ax.plot(xs, ys, zs, color='black', linewidth=2)

        for i, row in corners_df.iterrows():
            ax.text(row[0], row[1], row[2], str(i), fontsize=12, fontweight='bold')

        scatter = ax.scatter(tap_df[2], tap_df[3], tap_df[4], 
                        c=tap_df[1], 
                        cmap='tab10', 
                        s=15, 
                        label='Pressure Taps')
        

        seen_faces = set()

        # 3. Iterate through every tap
        for index, row in tap_df.iterrows():
            # Extract the data from the columns
            tap_id = row[0]
            face_num = row[1]
            x = row[2]
            y = row[3]
            z = row[4]
            
            # 4. Check if we have NEVER seen this face before
            if face_num not in seen_faces:
                # We found a completely new face! Plot the text.
                label_text = f"Face {int(face_num)}" 
                
                ax.text(x, y, z, label_text, 
                        color='black', 
                        fontsize=11, 
                        fontweight='bold',
                        ha='center', 
                        va='center',
                        bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black', pad=2))
                
                # 5. Add this face to our "seen" list so it NEVER prints again
                seen_faces.add(face_num)


        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Y (Length)')
        ax.set_zlabel('Z (Height)')
        ax.set_title('Building Wireframe Model with Face-Coded Taps')

        ax.set_box_aspect([100, 145, 40]) 
        ax.legend()

        plt.show()

    def get_wind_2d_plot(self, flat_frames_df, flat_corners_df, flat_tap_df):
        plt.figure(figsize=(10, 8))

        # Draw the lines connecting corners based on your pairs dataframe
        for _, row in flat_frames_df.iterrows():
            idx1 = int(row.iloc[0]) - 1 
            idx2 = int(row.iloc[1]) - 1

            
            # Pull the corresponding X and Y from your corners dataframe
            x_vals = [flat_corners_df.iloc[idx1, 0], flat_corners_df.iloc[idx2, 0]]
            y_vals = [flat_corners_df.iloc[idx1, 1], flat_corners_df.iloc[idx2, 1]]
            
            plt.plot(x_vals, y_vals, color='black', linewidth=1, alpha=0.7)


            plt.scatter(
            flat_tap_df[2], 
            flat_tap_df[3], 
            color='red',          # Make them stand out
            s=8,                 # Size of the dots
            marker='o',           # Shape (circle)
            label='Pressure Taps',
            zorder=3              # Ensures points stay ON TOP of the lines
            )


        unique_faces = flat_tap_df.iloc[:, 1].unique()

        # 2. Loop through each face one by one
        for face in unique_faces:
            # 3. Filter the dataframe to ONLY show taps on the current face
            face_data = flat_tap_df[flat_tap_df.iloc[:, 1] == face]

            # 4. Calculate the exact center point (Average X and Y)
            center_x = face_data.iloc[:, 2].mean()
            center_y = face_data.iloc[:, 3].mean()

            # 5. Place the label perfectly in the middle
            plt.text(center_x, center_y, f"Face {int(face)}", 
                    color='black', 
                    fontsize=12, 
                    fontweight='bold',
                    ha='center', 
                    va='center',
                    bbox=dict(facecolor='yellow', alpha=0.85, edgecolor='black', pad=3))
            


        plt.title("Flat Superimposed Tap Locations on Surface")
        plt.xlabel("X (Feet)")
        plt.ylabel("Y (Feet)")
        plt.gca().set_aspect('equal')
        plt.show()

    def get_mean_contour(self, face_no, flat_tap_coords,pressure_series):
        mean_cp_series = pressure_series.mean(axis = 0)

        pressure_df = mean_cp_series.reset_index()
        pressure_df.columns = ['Tap no.', 'mean_cp']
        
        
        all_means = pressure_df['mean_cp'].values
        num_taps = len(flat_tap_coords)
        matched_means = all_means[:num_taps]

        # 4. Paste it directly into the coordinate DataFrame
        flat_tap_coords['mean_cp'] = matched_means        
        
        face_dfs = {}

        # 2. Find all the unique face numbers in your dataset (e.g., [1.0, 2.0, 3.0...])
        # Using dropna() ensures we don't accidentally create a dataset for "NaN" faces
        unique_faces = flat_tap_coords[1].dropna().unique()

        # 3. Loop through and slice the master table
        for face_num in unique_faces:
            # Filter the master table for this specific face and create a clean copy
            face_data = flat_tap_coords[flat_tap_coords[1] == face_num].copy()
            
            # Store it in the dictionary using the face number as the key
            face_dfs[face_num] = face_data

        face_df = face_dfs[face_no]

        # Create a temporary DataFrame to safely drop duplicate spatial locations
        clean_df = face_df.drop_duplicates(subset=[2, 3])
        x = clean_df[2].values
        y = clean_df[3].values
        z = clean_df['mean_cp'].values

        # 2. Define the grid resolution
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        # 3. Interpolate the mean_cp values onto the grid

        
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # 4. Generate the contour plot
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label="Mean $C_p$")

        # Optional: Overlay the tap locations to verify grid coverage
        plt.scatter(x, y, c='black', s=15, marker='x', label='Taps')

        plt.title("Face chosen Mean Pressure Distribution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        # Grab your current minimums and maximums
        current_x_min, current_x_max = plt.xlim()
        current_y_min, current_y_max = plt.ylim()

        # Add a 5-unit "buffer" to all sides to zoom out
        buffer = 2.5 
        plt.xlim(current_x_min - buffer, current_x_max + buffer)
        plt.ylim(current_y_min - buffer, current_y_max + buffer)
        plt.show()


class TPU_DATA_ANALYZER:
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.df = None
    
    def get_loc_df(self):
        loc_matrix = self.data['Location_of_measured_points']
        loc_df = pd.DataFrame(loc_matrix.T, columns=['X', 'Y', 'Point_No', 'Face_No'])

        # THE FIX: If coordinates are stored in millimeters (max value > 5), 
        # convert them to standard meters automatically
        if loc_df['X'].max() > 1.0:
            print("⚠️ Detected millimeter units in coordinate matrix. Scaling to meters...")
            loc_df['X'] = loc_df['X'] / 1000.0
            loc_df['Y'] = loc_df['Y'] / 1000.0
        return loc_df
    
    def get_timeseries_df(self):
        pressure_df = pd.DataFrame(self.data['Wind_pressure_coefficients'])
        return pressure_df
    
    def get_channel_plot(self, loc_df, pressure_df):
        plt.figure(figsize=(12, 6))

        # 2. MATCH SIZES: Cap the location coordinates to the exact number of active channels
        total_active_channels = pressure_df.shape[1] # Number of columns in pressure matrix (200)
        active_loc_df = loc_df.head(total_active_channels).copy()

        # 3. Double-check column mapping if things look upside down:
        # If X and Y are swapped in the source matrix, flip them here:
        # x_coords = active_loc_df['Y'] 
        # y_coords = active_loc_df['X']
        x_coords = active_loc_df['X']
        y_coords = active_loc_df['Y']

        # Plot the active taps as blue crosses
        plt.scatter(x_coords, y_coords, marker='+', color='blue', s=100, linewidth=1)

        # 4. Iterate ONLY through the active capped dataframe
        for index, row in active_loc_df.iterrows():
            point_num = int(row['Point_No'])
            x_val = row['X']
            y_val = row['Y']
            
            # Offset the text slightly down and to the right so it doesn't overlap
            plt.text(x_val + 0.002, y_val - 0.002, str(point_num), 
                     color='black', fontsize=9, ha='left', va='top')

        # Add the vertical dashed lines separating sections

        # 1. Sort the taps horizontally from left to right
        sorted_loc = active_loc_df.sort_values(by='X')

        # 2. Track where the Face Number changes from one row to the next
        face_changes = sorted_loc['Face_No'].ne(sorted_loc['Face_No'].shift())
        
        # 3. Extract the exact X-coordinates where those shifts happen
        # We skip the very first point (index 0) because that's just the outer left wall
        boundary_xs = sorted_loc[face_changes]['X'].values[1:]

        # 4. Draw the dashed boundary lines exactly at the structural seam transitions
        for v_line in boundary_xs:

            snapped_line = round(v_line, 1)
            plt.axvline(x=snapped_line, color='blue', linestyle='--', alpha=0.5)

        # Lock the axes limits perfectly to the original geometry bounds
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        x_buffer = (x_max - x_min) * 0.1 if x_max != x_min else 0.05
        y_buffer = (y_max - y_min) * 0.1 if y_max != y_min else 0.05

        plt.xlim(max(0, x_min - x_buffer), x_max + x_buffer)
        plt.ylim(max(0, y_min - y_buffer), y_max + y_buffer)

        plt.title("Channels position", fontweight='bold', fontsize=12)
        plt.xlabel("Horizontal Direction /m", fontsize=11)
        plt.ylabel("Vertical Direction /m", fontsize=11)

        plt.tight_layout()
        plt.show()

    def mean_cp_contour(self, pressure_df, loc_df):
        # 1. Calculate the mean pressure coefficient (Cp) for every single channel
        mean_cp_series = pressure_df.mean(axis=0)
        mean_cp_df = mean_cp_series.reset_index()
        mean_cp_df.columns = ['Tap no.', 'mean_cp']

        all_means = mean_cp_df['mean_cp'].values
        num_taps = len(loc_df)
        matched_means = all_means[:num_taps]

        # Paste the calculated means directly into a copy of the coordinate DataFrame
        working_loc_df = loc_df.copy()
        working_loc_df['mean_cp'] = matched_means

        # =====================================================================
        # OPTION B INTEGRATION: DROP SPATIAL DUPLICATES TO PREVENT QHULL ERRORS
        # =====================================================================
        clean_df = working_loc_df.drop_duplicates(subset=['X', 'Y'])

        x = clean_df['X'].values
        y = clean_df['Y'].values
        z = clean_df['mean_cp'].values

        # =====================================================================
        # DYNAMIC GRID GENERATION (Scales automatically to any dataset size)
        # =====================================================================
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Create a dense grid mesh based on the absolute dimensions of the active data
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 100)
        )

        # Interpolate the scattered pressure points onto the dense coordinate grid mesh
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # =====================================================================
        # PLOTTING AND VISUALIZATION
        # =====================================================================
        plt.figure(figsize=(12, 6))

        # Plot the smooth color contour bands (RdBu_r = Classic Aerodynamic Red/Blue map)
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r', alpha=0.85)

        # Add the colorbar legend to the right side
        cbar = plt.colorbar(contour)
        cbar.set_label('Mean Pressure Coefficient ($C_p$)', fontsize=11)

        # Overlay the original tap locations as black crosses for visibility contrast
        plt.scatter(x, y, marker='+', color='black', s=60, linewidth=1)

        # Overlay the tap text labels
        for index, row in clean_df.iterrows():
            # Dynamic label offsets based on graph size metrics
            x_offset = (x_max - x_min) * 0.005
            y_offset = (y_max - y_min) * 0.015
            
            plt.text(row['X'] + x_offset, row['Y'] - y_offset, str(int(row['Point_No'])), 
                        color='black', fontsize=8, ha='left', va='top')

        # Dynamic Face Boundaries: Detect transitions and round to nearest decimal place
        sorted_loc = clean_df.sort_values(by='X')
        face_changes = sorted_loc['Face_No'].ne(sorted_loc['Face_No'].shift())
        boundary_xs = sorted_loc[face_changes]['X'].values[1:]

        for v_line in boundary_xs:
            snapped_line = round(v_line, 1)
            plt.axvline(x=snapped_line, color='black', linestyle='--', alpha=0.4)

        # Add Dynamic Bounding Box Margins (10% padding cushion)
        x_buffer = (x_max - x_min) * 0.1 if x_max != x_min else 0.05
        y_buffer = (y_max - y_min) * 0.1 if y_max != y_min else 0.05

        plt.xlim(max(0, x_min - x_buffer), x_max + x_buffer)
        plt.ylim(max(0, y_min - y_buffer), y_max + y_buffer)

        # Labels & Presentation
        plt.title("Global Mean Pressure Distribution with Channel Positions", fontweight='bold', fontsize=12)
        plt.xlabel("Horizontal Direction /m", fontsize=11)
        plt.ylabel("Vertical Direction /m", fontsize=11)

        plt.tight_layout()
        plt.show()



    def view_all_steps(self, pressure_df, tap_no):
        tap_idx = int(tap_no)

        pressure_at_tap = pressure_df[tap_idx]


        plt.plot(
            pressure_at_tap.index, 
            pressure_at_tap.values, 
            color='#1f77b4', 
            linewidth=0.5, 
            label=f'Tap Channel {tap_no}'
        )

        
        plt.title(f'Pressure Coefficient Time Series - Tap {tap_no}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Timestep (Frames / Samples)', fontsize=10)
        plt.ylabel('Pressure Value ($C_p$)', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')

        plt.tight_layout()

        plt.show()





    
