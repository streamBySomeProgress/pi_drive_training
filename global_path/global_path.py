import os

# 이미지와 관련하여 전역적으로 사용할 파일, 디렉터리 경로를 반환하는 코드들의 집합 파일
img_data_path = "./img_data"

# 학습된 모델과 관련하여 전역적으로 사용할 파일, 디렉터리 경로를 반환하는 코드들의 집합 파일
model_path = './torchArea/model/model.pth'

# 학습에 쓰이는 이미지와 해당 이미지를 분류하는 라벨의 연관관계를 작성한 파일
label_path = os.path.join(img_data_path, "labels.txt") # img_data_path 이하