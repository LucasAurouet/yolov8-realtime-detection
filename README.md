# YOLOv8 Real-Time Detection \& Fine-Tuning



This repository provides an **end-to-end pipeline** for **YOLOv8 object detection**:



- fine-tuning on a custom dataset.

- real-time inference using a webcam.

- GPU and FP16 inference support.



The project is designed as a **minimal working example** for experimenting with YOLOv8 rather than a production-ready application.



---



\## Repository Structure



```

yolov8-realtime-detection/

│

├── src/

│   ├── fine\_tune.py        # YOLOv8 loading and fine-tuning utilities 

│   └── video\_output.py    # # Real-time webcam inference with CUDA / FP16 support

│

├── notebooks/

│   ├── yolov8-finetune.ipynb   # Fine-tuning on a custom dataset (Kaggle)

│   └── yolov8-realtime.ipynb  # Real-time inference using trained weights

│

├── weights/

│   └── finetuned\_weights.pt    # Example fine-tuned model weights

│

├── requirements.txt

│

└── README.md

```



## Pipeline



### Fine-Tuning (Training)



Uses Ultralytics YOLOv8

Executed inside a Kaggle notebook

Trains on a custom dataset defined by a data.yaml file

Exports trained weights for later inference



### Real-Time Inference



Webcam capture via OpenCV

Frame-by-frame YOLOv8 inference

Bounding boxes, class IDs, and confidence scores displayed in real time

Optional CUDA + FP16 inference for better performance



## Source Files



fine\_tune.py



* Utilities for model loading and training
* Loads a pretrained YOLOv8 model (e.g. yolov8n)
* Fine-tunes the model on a custom dataset
* Saves the trained weights



get\_video.py



* Webcam stream reading
* YOLOv8 inference on each frame
* Optional bounding box rendering
* Simple and minimal implementation for experimentation



video\_output.py



* Loads fine-tuned YOLOv8 weights
* CUDA acceleration 
* YOLOv8 inference on each frame
* Optional bounding box rendering



## Notebooks (Kaggle)



yolov8-finetune.ipynb



* Fine-tunes YOLOv8 on a custom dataset 
* Exports finetuned\_weights.pt



yolov8-realtime.ipynb



* Loads fine-tuned weights
* Enables GPU and FP16 inference
* Runs real-time detection using the webcam stream



Kaggle is used for reproducibility and free GPU access, but is not mandatory for local usage.



## License



This project is provided for educational and experimental purposes.

YOLOv8 is subject to the Ultralytics license.

