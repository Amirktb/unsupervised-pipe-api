from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from unsupervised_model.pipeline import get_pipeline

def pipeline_trainer(data: pd.DataFrame, 
                     n_clusters: int) -> Tuple[Pipeline, pd.DataFrame, np.ndarray]:
    """
    Method to obtain a pipeline suitable for the uploaded 
    dataset and fit it on the dataset
    """
    
    pipe = get_pipeline(data=data, n_clusters=n_clusters)
    pipe.fit(data)
    processed_data = pipe.transform(data)
    predicted_labels = pipe["clusterer"]["kmeans_clusterer"].labels_
    
    return pipe, processed_data, predicted_labels
    