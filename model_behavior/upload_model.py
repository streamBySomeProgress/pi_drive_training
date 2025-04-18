import requests
import os
from requests.exceptions import RequestException
from dotenv import load_dotenv
from global_path.global_path import model_path

load_dotenv() # 환경변수 로드

# 변수 설정 (라즈베리 파이의 IP와 포트)
SERVER_IP = os.getenv('driving_server_ip')
SERVER_PORT = os.getenv('driving_server_port')

def uploadModel():
    """
        .pth 파일을 서버로 업로드하는 함수
        :return: 서버 응답
    """
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"file cannot be found: {model_path}")

    # 파일 확장자 검증
    if not model_path.endswith(".pth"):
        raise ValueError("file must have .pth extension")

    try:
        # 파일 전송
        with open(model_path, "rb") as f:
            files = {
                "file": (
                    os.path.basename(model_path),
                    f,
                    "application/octet-stream" # .pth 확장자에 대응하는 MIME 타입이 부재한 관계로 일반 바이너리 파일 타입의 content-type 사용
                )
            }
        response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/model/replace", files=files)


        # 응답 처리
        response.raise_for_status()  # 4xx, 5xx 에러 시 예외 발생
        return response.json()

    except RequestException as e:
        print(f"some error which is related network's problem occurred: {str(e)}")
        raise
    except ValueError as e:
        print(f"file validation error: {str(e)}")
        raise
    except Exception as e:
        print(f"unexpected error: {str(e)}")
        raise

# 해당 모듈을 타 영역에서 임포트할 시 본 코드와 같은 실행 영역은 삭제할 것
try:
    result = uploadModel()
    print("module is uploaded successfully:", result)
except Exception as e:
    print("uploading is failed:", str(e))