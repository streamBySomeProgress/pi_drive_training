import torch
import torch.nn as nn

# CNN 모델 정의
class LineCNN(nn.Module):
    def __init__(self):
        super(LineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 특징 맵 크기 축소(2 * 2 크기로 원본에서 2만큼(stride) 이동하며 downSample(2 * 2 맵에서의 최댓값(maxPool) -> one sample())
        self.fc1 = nn.Linear(32 * 120 * 160, 6)  # 32 * 120 * 160 의 입력 차원을 받아 6개 클래스 에 대응되는 6차원 요소로 반환
        self.dropout = nn.Dropout(0.5) # 과적합 방지를 위하여 일부 요소를 떨굼

    # 프레임워크에 의하여 굳이 명시하지 않더라도 backward 동작 또한 수행됨
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # 풀링 과정 (480x640 → 120x160)
        x = self.pool(torch.relu(self.conv2(x))) # 풀링 과정 (480x640 → 120x160)
        x = x.view(-1, 32 * 120 * 160) # 평탄화(텐서를 1차원 벡터로 변환 (614400 요소))
        x = self.dropout(x)
        x = self.fc1(x)
        return x