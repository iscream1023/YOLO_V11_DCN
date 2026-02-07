'''
#쿠다 확인
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch built with CUDA version: {torch.version.cuda}")
'''

from ultralytics import YOLO
model = YOLO("runs/detect/yolov11n_DCN/weights/best.pt")
model.info()

model = YOLO("yolo11n.pt")
model.info()

# 추론 테스트 (레이어 크기 확인용)
'''
import torch
dummy_input = torch.randn(1, 3, 640, 640)
result = model.predict(dummy_input)
'''

'''
#모델 테스트
from ultralytics import YOLO
model = YOLO("yolo_v11n_DCN.yaml")
model.info() 
'''
