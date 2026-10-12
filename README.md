# 🌔 Autonomous Lunar Landing Hazard Detection & Avoidance (Model Weights, 3D DEM & Benchmark Release)

Official model weights (`PyTorch` & `ONNX`), evaluation telemetry, coupled **Optical + 3D Digital Elevation Model (DEM)** test samples, and zero-shot **NASA Lunar Reconnaissance Orbiter (LRO)** validation benchmarks for real-time autonomous planetary descent and hazard avoidance (Artemis HLS / Chandrayaan / CLPS missions).

![Status](https://img.shields.io/badge/Artifact%20Release-Production%20Models%20%26%20Benchmarks-brightgreen)
![Model](https://img.shields.io/badge/Detector-YOLO11n-orange)
![Edge](https://img.shields.io/badge/Edge%20Runtime-ONNX%20%7C%20TensorRT-76b900)
![Validation](https://img.shields.io/badge/Zero--Shot%20Validation-NASA%20LRO%20%2F%20Apollo-0b3d91)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Executive Summary & Problem Statement

During terminal lunar descent, communication round-trip latency and orbital occlusion make Earth-based manual piloting impossible. Pre-mission orbital maps often lack the sub-meter resolution required to resolve local hazards such as fresh secondary craters, ejecta boulder aprons, and steep local scarps that can puncture a descent stage or cause catastrophic **spacecraft tip-over** upon touchdown.

This repository provides the **pre-trained neural weights, edge-compiled ONNX runtime graphs, coupled 3D DEM test samples, and validation telemetry** for a 3D-aware Guidance, Navigation, and Control (GNC) perception pipeline that:

1. **Models Coupled 3D DEM & Optical Terrains:** Trained on multi-octave $1/f$ fractal lunar elevation maps  Ground Sample Distance) with Fourier-perturbed elliptical craters following Pike's depth-to-diameter relation , rendered via Lambertian reflectance , ray-marched cast shadows, and stratigraphic occlusion pruning.
2. **Perceives Hazards Zero-Shot on Real Lunar Imagery:** Pairs a lightweight **YOLO11n** backbone with **CLAHE photometric normalization** to generalize zero-shot onto real **NASA LRO / Apollo** flight imagery without fine-tuning.
3. **Enforces 3D Mechanical Tip-Over Slope Limits:** Evaluates surface slope angles from 3D DEM heightmaps—or reconstructs a relative 3D DEM from monocular imagery via **Frankot-Chellappa FFT Photoclinometry (Shape-from-Shading)**—to mask out any terrain exceeding the **$10^\circ$ lander tip-over limit**.
4. **Optimizes Multi-Objective Touchdown Coordinates:** Fuses boundary-constrained **Euclidean Distance Transforms  with local slope penalties and lateral propellant divert costs to identify the safest, flattest, and most fuel-efficient touchdown zone.
5. **Executes at Flight-Computer Edge Speeds:** Ships with a simplified **ONNX Runtime (`opset=17`)** model achieving **P99 tail latencies under `11 ms`**, well inside a  powered-descent GNC cycle budget.

---

## 🏛️ Repository File System Architecture

This repository is structured around the exported artifacts, pre-trained weights, validation telemetry, and sample 3D DEM evaluation data:

```text
Lunar-Landing-Hazard-Avoidance/
│
├── README.md                                       # Main documentation & aerospace engineering report
│
├── models/                                         # Pre-trained production & edge flight weights
│   ├── lunar_yolo11n_best.pt                       # Trained PyTorch FP32 checkpoint (5.35 MB)
│   └── lunar_yolo11n_best.onnx                     # Simplified opset-17 Edge ONNX model (10.24 MB)
│
├── reports_and_plots/                              # Zero-shot NASA LRO validation & benchmark logs
│   ├── README_BENCHMARK.md                         # Auto-generated ID vs. OOD, LRO & ONNX latency tables
│   └── real_lro_dem_validation.png                 # 3x4 diagnostic grid: Real NASA LRO + 3D DEM + Slope + Safe Site
│
├── training_metrics/                               # Ultralytics training telemetry & evaluation curves
│   ├── results.png                                 # Box loss, cls loss, DFL loss & mAP progression curves
│   ├── results.csv                                 # Epoch-by-epoch quantitative training telemetry
│   ├── PR_curve.png                                # Precision-Recall curve across confidence thresholds
│   ├── F1_curve.png                                # F1-Score vs. confidence threshold curve
│   ├── P_curve.png                                 # Precision vs. confidence threshold curve
│   ├── R_curve.png                                 # Recall vs. confidence threshold curve
│   ├── confusion_matrix.png                        # Raw classification confusion matrix
│   ├── confusion_matrix_normalized.png             # Normalized classification confusion matrix
│   └── args.yaml                                   # Exact training hyperparameter configuration
│
└── sample_dataset/                                 # Coupled Optical + 3D DEM Out-of-Distribution (OOD) test samples
    ├── images/                                     # CLAHE-normalized optical frames (lunar_0000.jpg – lunar_0009.jpg)
    ├── dems/                                       # Co-registered 3D Digital Elevation Models in meters (lunar_0000.npy – lunar_0009.npy)
    └── labels/                                     # Occlusion-pruned ground-truth YOLO annotations (lunar_0000.txt – lunar_0009.txt)
```

---

## 🖼️ Visual Results: Zero-Shot Real NASA LRO Validation & 3D Slope Gating

![Zero-Shot Real LRO & 3D DEM Validation](reports_and_plots/real_lro_dem_validation.png)

*Figure 1: Zero-shot execution on real NASA lunar nadir imagery (`reports_and_plots/real_lro_dem_validation.png`). **Column 1:** CLAHE-normalized optical feed with `lunar_yolo11n_best` crater detections (red). **Column 2:** 3D Digital Elevation Model (DEM) in meters reconstructed via FFT Photoclinometry. **Column 3:** Local 3D surface slope map ($\theta_{\text{slope}}$ in degrees) enforcing a hard  spacecraft tip-over cutoff. **Column 4:** Multi-objective GNC touchdown solution (green inscribed clearance circle and white target crosshair).*

---

## 📈 Training Telemetry & Convergence Curves

| Training & Validation Convergence (`results.png`) | Precision-Recall Curve (`BoxPR_curve.png`) |
| :---: | :---: |
| ![Training Results](training_metrics/results.png) | ![PR Curve](training_metrics/BoxPR_curve.png) |

| Normalized Confusion Matrix (`confusion_matrix_normalized.png`) | F1-Confidence Harmonic Curve (`BoxF1_curve.png`) |
| :---: | :---: |
| ![Confusion Matrix](training_metrics/confusion_matrix_normalized.png) | ![F1 Curve](training_metrics/BoxF1_curve.png) |
---

## 🔬 System Workflow & Dual-Stream GNC Architecture

The architecture fuses deep neural object detection with deterministic 3D elevation derivatives so that an uncratered but steeply tilted hillside—or an undetected sub-meter scarp—never gets selected as a landing zone:

```text
       [ Nadir Optical Camera Feed (0.5 m/px GSD) ]
                            │
                            ▼
         [ CLAHE Photometric Normalization ]
              │                           │
              ▼                           ▼
  [ YOLO11n ONNX Backbone ]     [ 3D DEM / FFT Photoclinometry ]
  (Detects Crater Bounds)       (Computes Elevation Z(x,y) in m)
              │                           │
              ▼                           ▼
  [ Elliptical Footprint ]      [ Sobel 3D Slope Angle Map ]
  (Pads Craters by R_lander)    (Rejects Terrain > 10° Slope)
              │                           │
              └────────────►◄─────────────┘
                            │
                            ▼
       [ Boundary-Constrained Binary Safe Mask ]
       (0 = Crater / >10° Slope / FOV Edge, 255 = Traversable)
                            │
                            ▼
       [ Euclidean Distance Transform (L2 Clearance) ]
                            │
                            ▼
       [ Multi-Objective GNC Utility Optimization ]
       (Maximizes Clearance − Slope Penalty − Divert Fuel Cost)
                            │
                            ▼
       [ Optimal Safe Touchdown Coordinate (X*, Y*, R_safe) ]
```



## 📊 Performance & Comprehensive Benchmark Suite

*Full auto-generated benchmark tables are also available in [`reports_and_plots/README_BENCHMARK.md`](reports_and_plots/README_BENCHMARK.md) and epoch-by-epoch telemetry in [`training_metrics/results.csv`](training_metrics/results.csv).*

### 1. Synthetic Generalization: In-Distribution vs. Out-of-Distribution (OOD) Stress Test
Validation is split into an **In-Distribution split  and an **Out-of-Distribution Stress Test (`val_ood`)** featuring unseen grazing terminator shadows , heavier crater clustering, steeper terrain relief, and sensor read noise.

| Evaluation Split | mAP@50 | mAP@50-95 | Precision (P) | Recall (R) | Engineering Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **In-Distribution (`val_id`)** | `0.948` | `0.812` | `0.936` | `0.918` | High localization accuracy under nominal descent illumination ($14^\circ\text{–}45^\circ$ sun angle). |
| **Out-of-Distribution (`val_ood`)** | `0.874` | `0.695` | `0.889` | `0.842` | Strong resilience under unseen grazing terminator shadows, overlapping rims, and sensor noise. |

### 2. Zero-Shot Transfer on Real NASA Lunar Imagery (LRO / Apollo)
Evaluated **zero-shot** (no fine-tuning) across 6 real lunar surface regions at $512 \times 512$ resolution ($0.5\text{ m/px}$ nominal GSD):

| Real Lunar Region | Detected Hazards | Mean Conf | Safe Radius ($R_{\text{safe}}$) | Touchdown Slope | 98th Pct Scene Slope |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LRO-01: Daedalus Central Peak & Floor** | 14 | `0.612` | `28.5 m` | `3.18°` | `14.2°` |
| **LRO-02: Highland Degraded Crater Rim** | 19 | `0.584` | `21.0 m` | `4.05°` | `16.8°` |
| **LRO-03: Mare-Highland Contact Zone** | 11 | `0.645` | `34.0 m` | `2.42°` | `11.9°` |
| **LRO-04: Dense Secondary Crater Cluster** | 24 | `0.558` | `16.5 m` | `4.61°` | `17.5°` |
| **LRO-05: Grazing Terminator Shadow Field** | 17 | `0.531` | `22.5 m` | `3.89°` | `18.4°` |
| **LRO-06: Apollo Nadir Approach Corridor** | 12 | `0.629` | `31.5 m` | `2.74°` | `12.6°` |

### 3. Edge Hardware Latency Benchmark (PyTorch vs. ONNX Runtime)
Benchmarked over 80 timed iterations (after 15 warm-up passes) at $512 \times 512$ resolution on an NVIDIA Tesla T4 GPU:

| Runtime / Pipeline Stage | Weight Size | Mean Latency | P50 (ms) | P95 (ms) | P99 (ms) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Eager (`lunar_yolo11n_best.pt`)** | `5.35 MB` | `11.42 ms` | `11.10 ms` | `13.45 ms` | `15.20 ms` | `87.6 FPS` |
| **ONNX Runtime (`lunar_yolo11n_best.onnx`)** | `10.24 MB` | `6.18 ms` | `5.95 ms` | `7.12 ms` | `8.04 ms` | **`161.8 FPS`** |
| **Full GNC Loop (ONNX + 3D DEM Slope + $L_2$ Dist)** | `10.24 MB` | `8.45 ms` | `8.20 ms` | `9.65 ms` | `10.80 ms` | **`118.3 FPS`** |

---

## ⚡ Quick Start: Using the Released Weights & Sample Dataset

You can run inference directly using the pre-trained weights in `models/` on the included Out-of-Distribution lunar samples in `sample_dataset/`:

### 1. Run Crater Detection via Ultralytics CLI
```bash
pip install ultralytics onnxruntime opencv-python-headless numpy
yolo predict model=models/lunar_yolo11n_best.pt source=sample_dataset/images/ conf=0.25
```

### 2. Inspect a Coupled Optical + 3D DEM Sample (`sample_dataset/dems/`)
Each image in `sample_dataset/images/lunar_XXXX.jpg` is paired with a physical 3D elevation map in meters (`sample_dataset/dems/lunar_XXXX.npy`):
```python
import cv2
import numpy as np
from ultralytics import YOLO

# Load pre-trained ONNX or PyTorch model and sample 3D DEM
model = YOLO("models/lunar_yolo11n_best.onnx")
img = cv2.imread("sample_dataset/images/lunar_0000.jpg")
dem_meters = np.load("sample_dataset/dems/lunar_0000.npy")

# Run hazard detection
results = model.predict(img, conf=0.25, verbose=False)[0]
print(f"Detected {len(results.boxes)} craters | DEM Relief: {dem_meters.min():.1f}m to {dem_meters.max():.1f}m")
```

### 3. Compile `lunar_yolo11n_best.onnx` to TensorRT (NVIDIA Jetson Orin / Nano)
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/lunar_yolo11n_best.onnx \
    --saveEngine=models/lunar_yolo11n_fp16.engine \
    --fp16 \
    --workspace=1024
```

---

## 🛠️ Tech Stack

* **Model Formats:** PyTorch (`lunar_yolo11n_best.pt`), ONNX Runtime (`lunar_yolo11n_best.onnx`, `opset=17`)
* **3D Terrain Physics & GNC Mathematics:** OpenCV (`cv2.distanceTransform`, `cv2.Sobel`, `cv2.createCLAHE`), NumPy (`np.fft` Frankot-Chellappa Poisson Photoclinometry)
* **Telemetry & Reporting:** Matplotlib, Ultralytics YOLO11

---

## 👨‍💻 Author & Maintainer

**ASHISH TOMAR**  
*AI/ML Engineer & Aerospace Systems Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ashish%20Tomar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/ashish-tomar-/)
[![GitHub](https://img.shields.io/badge/GitHub-Tomar--Ashish--Singh-181717?style=flat&logo=github)](https://github.com/Tomar-Ashish-Singh/)
