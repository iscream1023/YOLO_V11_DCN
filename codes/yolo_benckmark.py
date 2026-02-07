from ultralytics.utils.benchmarks import benchmark

benchmark(model="/home/haggi/make_model/runs/detect/yolov11n_DCN_backbone/weights/best.pt", 
          data="/home/haggi/make_model/my_datasets/data.yaml", 
          imgsz=640, 
          half=False, 
          device=0, 
          )