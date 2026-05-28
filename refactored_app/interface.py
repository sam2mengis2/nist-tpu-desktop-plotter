from abc import ABC, abstractmethod

class TPU_Interface(ABC):

    @abstractmethod
    def get_channel_plot(self, loc_df, pressure_df):
        pass

    @abstractmethod
    def view_all_steps(self, tap_no):
        pass

    @abstractmethod
    def mean_cp_contour(self, pressure_df, loc_df, face_no):
        pass

    @abstractmethod
    def std_cp_contour(self, pressure_df, face_no):
        pass


class NIST_Interface(ABC):

    @abstractmethod
    def extract_dataframes(self):
        pass

    @abstractmethod
    def get_wind_frame_plot_3D(self, tap_df, frame_df, corners_df):
        pass

    @abstractmethod
    def view_full_series_tap(self, pressure_time_series, tap_no):
        pass

    @abstractmethod
    def get_wind_2d_plot(self, flat_frames_df, flat_corners_df, flat_tap_df):
        pass

    @abstractmethod
    def view_history_at_time(self, pressure_time_series, tap_no, time_a, time_b):
        pass

    @abstractmethod
    def get_mean_contour(self, face_no, flat_tap_coords, pressure_series):
        pass
    
    @abstractmethod
    def get_std_contour(self, face_no, flat_tap_coords, pressure_series):
        pass

    