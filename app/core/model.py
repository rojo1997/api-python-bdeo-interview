"""Model module"""
import torch


class Model:
    """Class to manage model loading and predictions"""

    def __init__(self, filename: str):
        self.filename = filename
        self.load_model(filename)

    def load_model(self, filename: str):
        """Load pre training model

        Args:
            filename (str): filename
        """
        self.model = torch.jit.load(filename)

    def predict(self, array: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            array (torch.Tensor): Feature vector

        Returns:
            torch.Tensor: Prediction
        """
        return self.model(array)
