import logging
import os, json
from logging.handlers import RotatingFileHandler
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
import uvicorn
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from app.routes import complaint_api

from app.routes.complaint_api import complaint_router
from app import mlesm_logger

mlesm_logger.info('Start ML Serving')

rootPath = os.environ.get('APP_ROOT_PATH', '/model')
app = FastAPI(root_path=rootPath,
              openapi_url='/openapi.json',
              docs_url=rootPath + '/docs',
              redoc_url=rootPath + '/docs',
              title="FASTAPI "+ os.environ.get('APPMODEL_NAME',
                                               'tfidf'))
app.include_router(complaint_router, prefix=rootPath + "/v1/predict")


@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(200))
    complaint_api.on_startup()


@app.get(rootPath + "/actuator/health", include_in_schema=False)
def get_path(request: Request):
    return {"message": "Healthy", "root_path": request.scope.get("root_path")}


@app.get(rootPath + "/openapi.json", include_in_schema=False)
async def openapi():
    print("got this")
    return get_openapi(title=app.title, version=app.version, routes=app.routes)

if __name__ == "__main__":
    portCfgDefault = { "port_num": 7072, "desc": " used for fastapi port."}
    portCfg = os.environ.get("PORT_CONFIG", portCfgDefault)
    if type(portCfg) == str:
        portCfg = json.loads(portCfg)
    uvicorn.run(app, host="0.0.0.0", port=int(portCfg["port_num"]))
    
    