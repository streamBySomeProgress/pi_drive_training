from fastapi import APIRouter, File, UploadFile, HTTPException
import os
import logging
from log.logger import setup_logger
from global_path.global_path import img_data_path, label_path


logging_info = setup_logger('main', 'log_upload_requestHandler.txt', logging.INFO)

upload_handler = APIRouter(prefix="/upload")

@upload_handler.post("/image")
async def upload_img(class_label: int, image: UploadFile = File(...)):
    """
    단일 이미지를 업로드하고 저장.
    - 이미지: JPEG 포맷.
    - 저장: images/image_YYYYMMDD_HHMMSS.jpg.
    """
    try:
        # MIME 타입 검증
        if image.content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Only JPEG images are supported")

        # 이미지 데이터 읽기
        content = await image.read()

        # 클래스별 디렉토리 생성 및 저장
        class_dir = os.path.join(img_data_path, f"class_{class_label}")
        file_list = os.listdir(class_dir)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        filename = f"frame_{len(file_list)}.jpg"
        filepath = os.path.join(class_dir, filename)

        # 파일 저장
        with open(filepath, "wb") as f:
            f.write(content)

        # labels.txt에 상대경로 및 이에 대응되는 라벨을 기록 (상대 경로 사용(class_0/frame_0.jpg 0))
        rel_filepath = os.path.relpath(filepath, img_data_path)
        with open(label_path, "a") as f:
            f.write(f"{rel_filepath} {class_label}\n")

        return {"status": "success", "filename": filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")