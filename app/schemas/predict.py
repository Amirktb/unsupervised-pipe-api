from typing import Optional
import pandas as pd
import numpy as np

from pydantic import BaseModel


class PredictionResults(BaseModel):
    predicted_labels: Optional[np.ndarray]
    processed_data: Optional[pd.DataFrame]
    version: str
