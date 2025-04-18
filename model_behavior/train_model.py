import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from torchArea.cnn.lineCnn import LineCNN
from global_path.global_path import img_data_path, model_path, label_path
from torch.utils.data import random_split
import requests
from dotenv import load_dotenv

load_dotenv() # 환경변수 로드

# 변수 설정 (학습을 수행하는 서버의 IP와 포트)
SERVER_IP = os.getenv('driving_server_ip')
SERVER_PORT = os.getenv('driving_server_port')

# 데이터셋 정의
class LineDataset(Dataset):
    # label_file 은 데이터 라벨
    def __init__(self, label_file, img_dir, augment = False):
        self.img_labels = [(line.split()[0], int(line.split()[1])) for line in open(label_file)] # "class_0/frame_0.jpg", 0" 형식의 tuple
        self.img_dir = img_dir
        # todo 증강 여부에 따른 결과 차이 관찰해 보기
        if augment:
            # 증강
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize(480),
                transforms.CenterCrop((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # 기존
            self.transform = transforms.Compose([
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB") # 이미지 디렉터리와 클래스 디렉터리(실 이미지 파일 포함) join
        image = self.transform(image)
        return image, label

# 모델, 손실 함수, 최적화
model = LineCNN()
criterion = nn.CrossEntropyLoss() # 다중 클래스 분류 영역에 적합한 손실 함수를 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델데이터셋 및 로더
dataset = LineDataset(label_path, img_data_path)
train_size = int(0.8 * len(dataset)) # 훈련, 검증에 쓰이는 데이터 비율은 8 : 2
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # 학습용
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) # 실 검증용

# 모델 훈련 함수
def trainModelWithEval():
    # 학습 루프
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 여기서부터 테스트(평가) 영역
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

    if os.path.exists(model_path):
        # 기존 모델 삭제
        os.remove(model_path)

    # 모델 저장
    torch.save(model.state_dict(), model_path) # 파일 형식으로 저장됨(모델의 state_dict만 저장)
    print(f"Model saved on {model_path}")

trainModelWithEval() # 본 모듈을 타 영역에서 임포트할 시 본 함수 호출 영역은 삭제