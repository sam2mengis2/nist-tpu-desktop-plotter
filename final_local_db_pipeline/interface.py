from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class TPUDataConverterInterface(ABC):
    """
    Abstract Base Class acting as the Interface for TPU data operations.
    Defines the contract that any data processor implementation must fulfill.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> None:
        """Load and extract arrays from the raw data source."""
        pass

    @abstractmethod
    def export_metadata(self, output_dir: str = "output") -> Optional[pd.DataFrame]:
        """Export sensor coordinates and building configuration metadata to CSV."""
        pass

    @abstractmethod
    def export_points(self, requested_points: List[int], output_dir: str = "output") -> Optional[pd.DataFrame]:
        """Export time-series data for a specific subset of user-requested pressure points."""
        pass

    @abstractmethod
    def export_by_surface(self, target_surface: int, output_dir: str = "output") -> Optional[pd.DataFrame]:
        """Export time-series data for all sensors matching a specific building face."""
        pass