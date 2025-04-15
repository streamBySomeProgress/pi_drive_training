from fastapi import FastAPI
import logging
from log.logger import setup_logger
from router.upload.upload_requestHandler import upload_handler
import uvicorn

app = FastAPI()

# 로깅 설정
logging_info = setup_logger('main', 'log_main.txt', logging.INFO)

app.include_router(upload_handler)

if __name__ == "__main__":
    logging_info.info("FastAPI server starting...")
    uvicorn.run(app, host="0.0.0.0", port=5000)