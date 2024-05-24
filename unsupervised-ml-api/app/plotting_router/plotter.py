from typing import Tuple, List, Any
import io
import base64

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

from app.trainer.unsupervised_trainer import pipeline_trainer
from app.data_router.csv_loader import api_router_loader

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi import  Request
from fastapi.templating import Jinja2Templates


api_router_plotter = APIRouter()
templates = Jinja2Templates(directory="app/templates")


class Clusterer:
    """
    Class to train and get processed data from the pipeline. 
    It also analyze the result of clustering with different
    number of clusters using Elbow methods and Silhouette
    coefficients.
    """
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 n_clusters: int) -> None:
        self.pipe, self.processed_data, predicted_labels = pipeline_trainer(
            data=dataset, 
            n_clusters=n_clusters
            )
        self.preprocessed_data = self.pipe["preprocessor"].transform(dataset)
        self.processed_data["predicted_labels"] = predicted_labels

    def _elbow_silhouette_analysis(self) -> Tuple[List, List]:
        kmeans_params = {
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        }
        ssd = []
        sil_coefs = []
        for k in range(2, 12):
            kmeans_clusterer = KMeans(n_clusters=k, 
                              **kmeans_params)
            kmeans_clusterer.fit(self.preprocessed_data)
            ssd.append(kmeans_clusterer.inertia_)

            score = silhouette_score(self.preprocessed_data, 
                                     kmeans_clusterer.labels_)
            sil_coefs.append(score)
            del score
        ssd = pd.DataFrame({"n": range(2, 12), "ssd": ssd})
        sil_coefs = pd.DataFrame({"n": range(2, 12), "silh": sil_coefs})
        return ssd, sil_coefs
    

def plot_scatter(processed_data: pd.DataFrame) -> Any:
    """
    Method to plot the two dimensionaly map of the 
    processed data.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=processed_data, x="pca0", y="pca1",
                    hue="predicted_labels", s=50, palette="Set2")
    
    ax.set_title("Two dimensional map of the data")

    # Save figure to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)

    # Encode figure data as base64 string
    fig_base64 = base64.b64encode(img_buf.getvalue())
    return fig_base64


def plot_elbow_method(ssd: List[float]) -> Any:
    """
    Method to plot the Sum of Squared distances vs. 
    the number of clusters.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=ssd, x="n", y="ssd", ax=ax)
    ax.set_title("Elbow Method Analysis")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Sum of Squared Distances")

    # Save figure to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)

    # Encode figure data as base64 string
    ssd_fig_base64 = base64.b64encode(img_buf.getvalue())
    return ssd_fig_base64


def plot_silh_coefs(sil_coefs: List[float]) -> Any:
    """
    Method to plot the Sum of Squared distances vs. 
    the number of clusters.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=sil_coefs, x="n", y="silh", ax=ax)
    ax.set_title("Silhouette Coefficients")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette Coefficient")

    # Save figure to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)

    # Encode figure data as base64 string
    silh_fig_base64 = base64.b64encode(img_buf.getvalue())
    return silh_fig_base64
    


@api_router_plotter.get("/analysis", response_class=HTMLResponse)
async def render_figure(request: Request) -> Any:
    """
    Plot the results of clustering and clustering analysis.
    """

    clusterer = Clusterer(dataset=api_router_loader.dataset,
                      n_clusters=api_router_loader.n_clusters)
    ssd, sil_coefs = clusterer._elbow_silhouette_analysis()

    ssd_fig_base64 = plot_elbow_method(ssd)
    silh_fig_base64 = plot_silh_coefs(sil_coefs)
    result_fig_base64 = plot_scatter(clusterer.processed_data)

    details_dict = {"request": request, 
                    "figure_base64_elb": ssd_fig_base64.decode('utf-8'),
                    "figure_base64_silh": silh_fig_base64.decode('utf-8'),
                    "figure_base64_result": result_fig_base64.decode('utf-8'),
                    "n_clusters": api_router_loader.n_clusters}

    return templates.TemplateResponse("fig_page.html", details_dict)
