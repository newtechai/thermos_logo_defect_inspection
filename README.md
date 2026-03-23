# YOLOv10 + VanillaNet-Style Backbone for logo Defect Detection

This repository extends the standard YOLOv10 architecture by integrating a **VanillaNet-inspired backbone** for improved defect detection, particularly for logo defect detection.

---

## Overview

The proposed model replaces the default YOLOv10 backbone with a structured backbone composed of:

- Depthwise separable convolutions  
- ELU activation functions  
- Squeeze-and-Excitation (SE) channel attention  
- SPPF-based multi-scale context aggregation  



---

## Key Contributions

- Custom **VanillaNet-style backbone** integrated into YOLOv10  
- Improved representation of subtle and small-scale defects  
- Efficient architecture with controlled computational cost  
- Full compatibility with standard YOLOv10 training and inference  

---

### Backbone Components

- `VanillaStem`  
- `VanillaStage`  
- `VanillaBlock`  
- `SELayer`  
- `VanillaSPPF`  

The backbone extracts hierarchical feature maps (P3, P4, P5) used by the standard YOLOv10 detection head.

---

## Installation

Clone YOLOv10 and install dependencies:

```bash
git clone https://github.com/newtechai/thermos_logo_defect_inspection.git
cd yolov10
pip install -r requirements.txt
pip install -e 

```
## Training
```bash
yolo detect train \
  model=ultralytics/cfg/models/v10/yolov10n_vanillanet.yaml \
  data=data.yaml \
  epochs=300 \
  imgsz=640 \
  batch=16 \
  device=0
```
---

## Citation

This work builds upon:

- **YOLOv10** – [https://arxiv.org/abs/2405.14458](https://arxiv.org/abs/2405.14458)  
- **VanillaNet** – [https://arxiv.org/abs/2305.12972](https://arxiv.org/abs/2305.12972)  