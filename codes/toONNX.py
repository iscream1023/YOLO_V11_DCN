from ultralytics import YOLO
import torch.nn as nn
import torch

model = YOLO('yolo_v11n.yaml')

#netron에서 보기 쉽게 onnx로 변환

model.export(format = 'ONNX')

#모델 형태 찍기
#print(model)

#CTRL+F 해서 SPPF 위치 찾으면 거기까지가 backbone
#지금 yolov11n은 (9)까지가 backbone

'''
#backbone 자르기
SPPF_pos = 9
backbone_layer = []
for m in model.model.model:
    backbone_layer.append(m)
    if isinstance(m,type(model.model.model[SPPF_pos])):
        print(f"SPPF 발견 : {SPPF_pos}")
        break
mybackbone = nn.Sequential(*backbone_layer)
print("자른 backbone을 onnx로 변환")
'''
dummy_input = torch.randn(1,3,640,640)
torch.onnx.export(
    model,
    dummy_input,
    "v11n_DCN.onnx",
    input_names=['input'],
    output_names=['output']
)
print("done!")