from pathlib import Path

import pandas as pd

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from fastapi import UploadFile, File, status, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


api_router_loader = APIRouter()

templates = Jinja2Templates(directory="templates")
api_router_loader.mount("/static",
           StaticFiles(
               directory=Path(__file__).parent.parent.absolute() / "static"), 
               name="static")

# Initializing the dataset as None
api_router_loader.n_clusters = None
api_router_loader.dataset = None


@api_router_loader.post("/upload-csv", response_class=RedirectResponse, status_code=200)
def upload_csv(file: UploadFile = File(...), 
               n_clusters: int = Form(...)) -> RedirectResponse:
    """
    Method to upload the file and read it as a 
    csv file in pandas dataframe.
    """

    # add n_clusters and datasets as properties to api_router_loader
    api_router_loader.n_clusters = n_clusters

    # raise an error if no file is uploaded
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded!")
    try:
        # Read the csv file with pandas
        api_router_loader.dataset = pd.read_csv(file.file)
        file.file.close()
    except Exception as e:
        raise HTTPException(status_code=400, 
                            detail=f"Error processing file {file.filename}: {str(e)}")
    
    return RedirectResponse(url="analysis", status_code=status.HTTP_303_SEE_OTHER)
