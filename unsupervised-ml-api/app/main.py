from typing import Any
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings, setup_app_logging
from app.data_router.csv_loader import api_router_loader
from app.plotting_router.plotter import api_router_plotter


# setup logging
setup_app_logging(config=settings)

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
app.mount("/static",
          StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
          name="static")
templates = Jinja2Templates(directory="app/templates")

root_router = APIRouter()


@root_router.get("/", response_class=HTMLResponse, status_code=200)
def index(request: Request) -> Any:
    """Main page html response."""
    return templates.TemplateResponse("index.html", {"request": request})


app.include_router(root_router)
app.include_router(api_router_loader)
app.include_router(api_router_plotter)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    logger.warning("Running in debugging mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
