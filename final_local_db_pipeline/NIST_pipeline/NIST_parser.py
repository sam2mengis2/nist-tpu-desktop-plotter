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
    def __init__(self):
        pass


    def get_mean_contour(self, face_no, flat_tap_coords, pressure_series):
        """Generates a mean pressure coefficient contour map. 
        Compatible with both legacy local files and pre-computed cloud database DataFrames.
        """
        # =========================================================================
        # 1. DATA EXTRACTION LAYER (Handles Cloud DB vs Legacy File structures)
        # =========================================================================
        
        # Check if the inputs match the new Cloud DB master dataframe properties
        is_cloud_data = False
        if isinstance(flat_tap_coords, pd.DataFrame):
            if 'x_coordinate' in flat_tap_coords.columns or 'x_coordinate' in flat_tap_coords.index:
                is_cloud_data = True

        if is_cloud_data:
            # If the dataframe was passed in transposed via the UI cache dictionary
            if 'x_coordinate' in flat_tap_coords.index:
                df_spatial = flat_tap_coords.T
                df_pressure = pressure_series.T
                
                # Extract values directly (Cloud data has pre-computed metrics)
                x = pd.to_numeric(df_spatial['x_coordinate']).values
                y = pd.to_numeric(df_spatial['y_coordinate']).values
                z = pd.to_numeric(df_pressure['mean_cp']).values
            else:
                # If the raw master_df was passed directly into the method
                clean_df = flat_tap_coords.drop_duplicates(subset=['x_coordinate', 'y_coordinate'])
                x = pd.to_numeric(clean_df['x_coordinate']).values
                y = pd.to_numeric(clean_df['y_coordinate']).values
                z = pd.to_numeric(clean_df['mean_cp']).values
                
        else:
            # Legacy local file path logic (.hdf / .mat)
            mean_cp_series = pressure_series.mean(axis=0)
            pressure_df = mean_cp_series.reset_index()
            pressure_df.columns = ['Tap no.', 'mean_cp']
            
            all_means = pressure_df['mean_cp'].values
            num_taps = len(flat_tap_coords)
            matched_means = all_means[:num_taps]
            
            flat_tap_coords = flat_tap_coords.copy()
            flat_tap_coords['mean_cp'] = matched_means        
            
            # Slicing the structural face layout
            unique_faces = flat_tap_coords[1].dropna().unique()
            face_dfs = {face_num: flat_tap_coords[flat_tap_coords[1] == face_num].copy() for face_num in unique_faces}
            
            face_df = face_dfs.get(face_no, list(face_dfs.values())[0])
            clean_df = face_df.drop_duplicates(subset=[2, 3])
            
            x = clean_df[2].values
            y = clean_df[3].values
            z = clean_df['mean_cp'].values

        # =========================================================================
        # 2. MATHEMATICAL SURFACE INTERPOLATION & PLOTTING LAYER
        # =========================================================================
        # Remove any potential NaN values to ensure griddata processing doesn't fail
        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

        # Define the high-resolution grid layout space
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        # Interpolate scatter array values onto the continuous coordinates grid mesh
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # Generate the colored contour canvas
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label="Mean $C_p$")

        # Overlay geometric tap coordinates to check coverage verification
        plt.scatter(x, y, c='black', s=15, marker='x', label='Taps')

        plt.title(f"Mean Pressure Coefficient ($C_p$) Distribution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        # Apply spatial bounding display safety padding
        current_x_min, current_x_max = plt.xlim()
        current_y_min, current_y_max = plt.ylim()
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

 