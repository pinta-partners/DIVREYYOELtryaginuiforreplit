import os
import logging
from fastapi import FastAPI
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient

from ..processing.processing import SearchHandler
from ..retrieval.datasources.vector_datasource import HybridEmbeddingsColDatasource
from ..retrieval.datasources.other_texts_datasource import OtherTextsDatasource
from ..models.models import QueryRequest, QueryResponse

# Initialize FastAPI app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How to run:
# 1. Start the FastAPI server by running: `hypercorn chassidic-ai.serving.server:app --reload --host 0.0.0.0 --port 8000`
# 2. Go to http://localhost:8000/docs to see the Swagger UI and test the API
# 3. Use the `/cards` endpoint to query for card data

# Ensure that MONGO_URI is set in the environment
if not os.getenv("MONGO_URI"):
    raise Exception("MONGO_URI environment variable not set")
# Ensure that OPENAI_API_KEY is set in the environment
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("OPENAI_API_KEY environment variable not set")
# Ensure that ANTHROPIC_API_KEY is set in the environment
if not os.getenv("ANTHROPIC_API_KEY"):
    raise Exception("ANTHROPIC_API_KEY environment variable not set")


@app.on_event("startup")
async def startup_event():
    mongo_connection_string = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_client = AsyncIOMotorClient(host=mongo_connection_string)
    app.state.embedding_datasource = await HybridEmbeddingsColDatasource.create(
        mongo_client
    )
    app.state.other_texts_datasource = await OtherTextsDatasource.create(mongo_client)


@app.post("/question", response_model=QueryResponse)
async def post_question(request: QueryRequest):
    # query = request.query
    # response_uuid = str(uuid.uuid4())

    search_handle = SearchHandler(
        embedding_datasource=app.state.embedding_datasource,
        other_texts_datasource=app.state.other_texts_datasource,
    )
    response = await search_handle.search(request, top_k=20)
    return response


# Serve the HTML in `ui.html`
@app.get("/")
async def root():
    return FileResponse("static/ui.html")


# Solve CORS
@app.middleware("http")
async def add_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
