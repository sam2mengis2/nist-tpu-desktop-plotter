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
from interface import TPU_Interface


class TPU_High_Rise_Analyzer(TPU_Interface):
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

    def mean_cp_contour(self, pressure_df, loc_df, face_no):
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

    def get_std_contour(self, face_no, flat_tap_coords, pressure_series):
        return False



class TPU_Adj_Analyzer(TPU_Interface):
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
        return False
    
    def view_all_steps(self, tap_no):
        return False
    
    def mean_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    
    def std_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    

class TPU_NO_EAVE(TPU_Interface):
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
        return False
    
    def view_all_steps(self, tap_no):
        return False
    
    def mean_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    
    def std_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    

class TPU_WITH_EAVE(TPU_Interface):
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
        return False
    
    def view_all_steps(self, tap_no):
        return False
    
    def mean_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    
    def std_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    
class TPU_LOW_RISE(TPU_Interface):
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
        return False
    
    def view_all_steps(self, tap_no):
        return False
    
    def mean_cp_contour(self, pressure_df, loc_df, face_no):
        return False
    
    def std_cp_contour(self, pressure_df, loc_df, face_no):
        return False


    