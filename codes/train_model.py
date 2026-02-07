from ultralytics import YOLO
import torch
from torchvision import models,datasets,transforms

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
    print(f"@Warning@ cuda is unavailable @Warning@")
    exit(1)

model = YOLO('yolo_v11n_DCN_in_head.yaml') 
model.load('yolo11n.pt') 
model.to('cuda')

origin_datasets_root = r"/home/haggi/make_model/my_datasets/data.yaml"
small_datasets_root = r"/home/haggi/make_model/small_datasets/mini_data.yaml"

def main():
    model.train(
        data=origin_datasets_root,
        lr0=5e-4,
        project="yolo11n_DCN_head_large",
        epochs=40,
        workers=4,
        imgsz=640,
    )

if __name__ == "__main__":
    main()
    