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
from interface import NIST_Interface


class NIST_DATA_ANALYZER(NIST_Interface):
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

    def get_std_contour(self, face_no, flat_tap_coords, pressure_series):
            # 1. Calculate the standard deviation (std) across each column channel
            std_cp_series = pressure_series.std(axis=0)

            pressure_df = std_cp_series.reset_index()
            pressure_df.columns = ['Tap no.', 'std_cp']
            
            all_stds = pressure_df['std_cp'].values
            num_taps = len(flat_tap_coords)
            matched_stds = all_stds[:num_taps]

            # 2. Safely copy the coordinates and paste the std values into it
            # Making a copy prevents overwriting the columns used by your mean plots
            coords_copy = flat_tap_coords.copy()
            coords_copy['std_cp'] = matched_stds        
            
            face_dfs = {}

            # 3. Find all the unique face numbers in your dataset
            unique_faces = coords_copy[1].dropna().unique()

            # 4. Loop through and slice the master table
            for face_num in unique_faces:
                face_data = coords_copy[coords_copy[1] == face_num].copy()
                face_dfs[face_num] = face_data

            face_df = face_dfs[face_no]

            # Create a temporary DataFrame to safely drop duplicate spatial locations
            clean_df = face_df.drop_duplicates(subset=[2, 3])
            x = clean_df[2].values
            y = clean_df[3].values
            z = clean_df['std_cp'].values

            # 5. Define the grid resolution
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 100),
                np.linspace(y.min(), y.max(), 100)
            )

            # 6. Interpolate the std_cp values onto the grid
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

            # 7. Generate the contour plot
            plt.figure(figsize=(8, 6))
            
            # Using YlOrRd since standard deviations are always positive numbers
            contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='YlOrRd')
            
            # Add the formatted colorbar scale
            import matplotlib.ticker as ticker
            cbar = plt.colorbar(contour, label="Std Dev $C_p$")
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Overlay the tap locations to verify grid coverage
            plt.scatter(x, y, c='black', s=15, marker='x', label='Taps')

            plt.title(f"Face {int(face_no)} Standard Deviation Pressure Distribution", fontweight='bold')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()

            # Grab your current minimums and maximums
            current_x_min, current_x_max = plt.xlim()
            current_y_min, current_y_max = plt.ylim()

            # Keeps your identical 2.5 unit layout zoom-out buffer shift
            buffer = 2.5 
            plt.xlim(current_x_min - buffer, current_x_max + buffer)
            plt.ylim(current_y_min - buffer, current_y_max + buffer)
            
            plt.show()

 