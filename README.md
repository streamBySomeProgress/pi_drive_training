# pi_drive_training
라즈베리 파이를 사용한 주행을 위한 학습 로직을 작성하는 영역

가상환경 활성: source venv/bin/activate

비활성화: deactive

패키지 목록 최신화: pip freeze > requirements.txt

목록에 있는 패키지 일괄 설치: pip install -r requirements.txt

## 유의사항
1. 학습해야 할 클래스는 총 6개(Left, Center, Right, None, Left Curve, Right Curve)
2. 클래스당 최소 20~30장의 사진이 필요하며 각 클래스당 사진의 개수는 비슷하게 유지할 것(클래스당 최소 20개의 사진이 존재하지 않을 경우 샘플링 이외 동작을 수행하지 않음)
3. 학습을 위한 카메라 데이터는 라즈베리파이, 외부 환경 양쪽에서 받을 수 있도록 구현하기
