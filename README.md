# 🌔 Autonomous Lunar Landing Hazard Detection & Avoidance (3D DEM & Edge GNC Edition)

An end-to-end **Computer Vision, 3D Digital Elevation Model (DEM) Slope Estimation, and Spatial Guidance, Navigation, and Control (GNC)** pipeline designed for real-time autonomous planetary descent and hazard avoidance (NASA Artemis HLS / ISRO Chandrayaan / CLPS missions).

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Detector-YOLO11n-orange)
![Edge](https://img.shields.io/badge/Edge%20Runtime-ONNX%20%7C%20TensorRT-76b900)
![Validation](https://img.shields.io/badge/Zero--Shot%20Validation-NASA%20LRO%20%2F%20Apollo-0b3d91)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Executive Summary & Problem Statement

During terminal lunar descent, communication round-trip latency and orbital occlusion make Earth-based manual piloting impossible. Pre-mission orbital maps often lack the sub-meter resolution required to resolve local hazards such as fresh secondary craters, ejecta boulder aprons, and steep local scarps that can puncture a descent stage or cause catastrophic **spacecraft tip-over** upon touchdown.

Unlike simplified 2D bounding-box scripts that rely on synthetic circles and ignore terrain tilt, this repository implements a **physically grounded, 3D-aware GNC perception and planning pipeline** that:

1. **Synthesizes Coupled 3D DEM & Optical Terrains:** Generates multi-octave $1/f$ fractal lunar elevation maps ($0.5\text{ m/px}$ Ground Sample Distance), carves Fourier-perturbed elliptical craters following Pike's depth-to-diameter relation ($d/D \approx 0.20$), and renders nadir optical frames via **Lambertian reflectance** ($\mathbf{n} \cdot \mathbf{s}$) and ray-marched cast shadows with pixel-level stratigraphic occlusion tracking.
2. **Perceives Hazards Across Lighting Regimes:** Trains a lightweight **YOLO11n** detector paired with **CLAHE photometric normalization** to achieve zero-shot transferability onto real **NASA Lunar Reconnaissance Orbiter (LRO) / Apollo** flight imagery without fine-tuning.
3. **Enforces 3D Mechanical Tip-Over Slope Limits:** Evaluates surface slope angles $\theta_{\text{slope}}(x, y)$ directly from co-registered 3D DEM heightmaps—or reconstructs a relative 3D DEM from monocular imagery via **Frankot-Chellappa FFT Photoclinometry (Shape-from-Shading)**—to reject any terrain exceeding the **$10^\circ$ lander tip-over limit**.
4. **Optimizes Multi-Objective Touchdown Coordinates:** Fuses boundary-constrained **Euclidean Distance Transforms ($L_2$)** with local slope penalties and lateral propellant divert costs to select the safest, flattest, and most fuel-efficient landing zone in real time.
5. **Executes at Edge Flight-Computer Speeds:** Exports weights to a simplified **ONNX Runtime (`opset=17`)** graph, benchmarking **P50 / P95 / P99 tail latencies** well inside a $20\text{ Hz}$ ($50\text{ ms}$) powered-descent GNC cycle budget.

---

## 🏛️ Repository Architecture

```text
Lunar-Landing-Hazard-Avoidance/
│
├── lunar_hazard_avoidance.py               # Complete 6-phase training, 3D DEM GNC, LRO validation & ONNX pipeline
├── README.md                               # Project documentation & aerospace engineering report
│
├── models/
│   ├── lunar_yolo11n_best.pt               # Trained PyTorch FP32 checkpoint
│   └── lunar_yolo11n_best.onnx             # Simplified opset-17 Edge ONNX model for Jetson / ORT deployment
│
├── reports_and_plots/
│   ├── README_BENCHMARK.md                 # Auto-generated ID vs. OOD, Real LRO & ONNX latency tables
│   └── real_lro_dem_validation.png         # 3x4 diagnostic grid: Real NASA LRO + 3D DEM + Slope + Safe Site
│
├── training_metrics/
│   ├── results.png                         # Box loss, cls loss, DFL loss & mAP progression curves
│   ├── results.csv                         # Epoch-by-epoch training and validation telemetry
│   ├── PR_curve.png                        # Precision-Recall curve across confidence thresholds
│   └── confusion_matrix.png                # Classification confusion matrix
│
└── sample_dataset/
    ├── images/                             # Sample Out-of-Distribution (OOD) terminator optical frames (.jpg)
    ├── dems/                               # Coupled 3D Digital Elevation Models in meters (.npy)
    └── labels/                             # Occlusion-pruned ground-truth YOLO annotations (.txt)
```

---

## 🖼️ Visual Results: Real NASA Lunar Imagery + 3D DEM Slope Gating

![Zero-Shot Real LRO & 3D DEM Validation](reports_and_plots/real_lro_dem_validation.png)

*Figure 1: Zero-shot execution on real NASA lunar nadir imagery. **Column 1:** CLAHE-normalized optical feed with YOLO11n crater detections (red). **Column 2:** 3D Digital Elevation Model (DEM) in meters reconstructed via FFT Photoclinometry. **Column 3:** Local 3D surface slope map ($\theta_{\text{slope}}$ in degrees) enforcing a hard $\le 10^\circ$ spacecraft tip-over cutoff. **Column 4:** Multi-objective GNC touchdown solution (green inscribed clearance circle and white target crosshair).*

---

## 🔬 System Workflow & Dual-Stream GNC Architecture

The pipeline fuses deep neural object detection with deterministic 3D elevation derivatives so that an uncratered but steeply tilted hillside—or an undetected sub-meter scarp—never gets selected as a landing zone:

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

---

## 🧠 Technical Deep Dive

### 1. Coupled 3D DEM Synthesis & Domain Randomization
To eliminate the classic "synthetic circle bias" where models memorize flat 2D OpenCV circles and fail on real orbital imagery:
* **Multi-Octave $1/f$ Fractal Topography:** Regional mare and highland undulations are synthesized as continuous 3D heightmaps $Z(x, y)$ in meters across four spatial frequency octaves.
* **Fourier-Perturbed Impact Geometry:** Crater rims are parameterized in polar coordinates using randomized radial harmonics before elliptical projection:
  $$r(\theta) = r_0 \left(1 + \sum_{k=2}^{4} a_k \cos(k\theta + \phi_k)\right)$$
  Crater cavities follow **Pike's lunar depth-to-diameter relation** ($d/D \approx 0.20$) with parabolic bowls and continuous raised ejecta blankets ($1.0 < r/R \le 1.45$).
* **Lambertian Reflectance & Ray-Marched Shadows:** Optical images are rendered directly from the 3D DEM surface normals $\mathbf{n}(x, y)$ and solar incidence vector $\mathbf{s}(\alpha_{\text{az}}, \theta_{\text{el}})$, accompanied by step-wise horizon ray-casting for grazing terminator shadows ($\theta_{\text{el}} \in [5^\circ, 45^\circ]$).
* **Stratigraphic Occlusion Accounting:** An integer `instance_map` buffer tracks pixel overwrites when younger impacts obliterate older crater rims. Ground-truth bounding boxes are recomputed strictly from surviving visible pixels, pruning heavily obliterated craters ($<35\%$ visibility) to eliminate "ghost" labels.

### 2. 3D Surface Slope Estimation & Monocular Photoclinometry
A 2D object detector only bounds horizontal crater rims; it cannot detect whether an uncratered hillside is tilted at $18^\circ$.
* **Physical Slope Derivative:** Given the elevation grid $Z(x, y)$ and Ground Sample Distance ($\text{GSD} = 0.5\text{ m/px}$), true surface slope is computed via finite-difference Sobel operators:
  $$\theta_{\text{slope}}(x, y) = \arctan\left(\sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2}\right) \cdot \frac{180}{\pi}$$
* **Frankot-Chellappa FFT Photoclinometry:** When evaluating monocular 2D NASA LRO images without co-registered LOLA LiDAR data, the pipeline estimates the dominant solar azimuth from image irradiance gradients and integrates the surface gradient field $(p, q)$ in the Fourier domain to reconstruct a relative 3D heightmap $Z_{\text{est}}(x, y)$.

### 3. Multi-Objective GNC Landing Site Solver
First, the binary `safe_mask` zeroes out the 1-pixel camera frame perimeter (preventing the landing footprint from spilling outside the sensor's field of view into unmapped terrain), zeroes out all pixels where $\theta_{\text{slope}}(x, y) > 10^\circ$, and masks out elliptical YOLO crater predictions padded by the lander's physical radius ($R_{\text{lander}} = 7.0\text{ m} = 14\text{ px}$).

Next, `cv2.distanceTransform` computes the exact Euclidean clearance $D_{L_2}(x, y)$ to the nearest hazard:
$$D_{L_2}(x, y) = \min_{(x_0, y_0) \in \text{Hazards}} \sqrt{(x - x_0)^2 + (y - y_0)^2}$$

Finally, the GNC planner selects $(x^*, y^*) = \arg\max U(x, y)$ across the composite utility surface:
$$U(x, y) = \frac{D_{L_2}(x, y)}{D_{\max}} - w_{\text{slope}}\left(\frac{\theta_{\text{slope}}(x, y)}{\theta_{\max}}\right) - w_{\text{divert}}\left(\frac{\Vert{}(x, y) - (x_c, y_c)\Vert{}_2}{R_{\text{fov}}}\right)$$
subject to $D_{L_2}(x, y) \ge R_{\text{lander}}$. This simultaneously maximizes obstacle clearance, minimizes touchdown tilt, and conserves reaction-control / descent propellant by penalizing large lateral diverts away from the nominal boresight $(x_c, y_c)$.

---

## 📊 Performance & Comprehensive Benchmark Suite

### 1. Synthetic Generalization: In-Distribution vs. Out-of-Distribution (OOD) Stress Test
Rather than reporting inflated metrics on a single synthetic distribution, validation is split into an **In-Distribution split (`val_id`)** ($14^\circ\text{–}45^\circ$ sun elevation) and an **Out-of-Distribution Stress Test (`val_ood`)** featuring unseen grazing terminator shadows ($5^\circ\text{–}11.5^\circ$), higher crater clustering density, higher terrain relief, and sensor read noise.

| Evaluation Split | mAP@50 | mAP@50-95 | Precision ($P$) | Recall ($R$) | Engineering Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **In-Distribution (`val_id`)** | `0.948` | `0.812` | `0.936` | `0.918` | High localization accuracy on nominal descent illumination ($14^\circ\text{–}45^\circ$ sun angle). |
| **Out-of-Distribution (`val_ood`)** | `0.874` | `0.695` | `0.889` | `0.842` | Strong resilience under unseen grazing terminator shadows, overlapping rims, and sensor noise. |

### 2. Zero-Shot Transfer on Real NASA Lunar Imagery (LRO / Apollo)
Trained **exclusively on procedural 3D DEM data**, the model was evaluated **zero-shot** (no fine-tuning) across 6 real lunar surface regions at $512 \times 512$ resolution ($0.5\text{ m/px}$ nominal GSD):

| Real Lunar Region | Detected Hazards | Mean Conf | Safe Radius ($R_{\text{safe}}$) | Touchdown Slope | 98th Pct Scene Slope |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LRO-01: Daedalus Central Peak & Floor** | 14 | `0.612` | `28.5 m` | `3.18°` | `14.2°` |
| **LRO-02: Highland Degraded Crater Rim** | 19 | `0.584` | `21.0 m` | `4.05°` | `16.8°` |
| **LRO-03: Mare-Highland Contact Zone** | 11 | `0.645` | `34.0 m` | `2.42°` | `11.9°` |
| **LRO-04: Dense Secondary Crater Cluster** | 24 | `0.558` | `16.5 m` | `4.61°` | `17.5°` |
| **LRO-05: Grazing Terminator Shadow Field** | 17 | `0.531` | `22.5 m` | `3.89°` | `18.4°` |
| **LRO-06: Apollo Nadir Approach Corridor** | 12 | `0.629` | `31.5 m` | `2.74°` | `12.6°` |

### 3. Edge Hardware Latency Benchmark (PyTorch vs. ONNX Runtime)
Benchmarked over 80 timed iterations (following 15 warm-up passes) at $512 \times 512$ input resolution on an NVIDIA Tesla T4 GPU:

| Runtime / Pipeline Stage | Weight Size | Mean Latency | P50 (ms) | P95 (ms) | P99 (ms) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Eager (`best.pt`)** | `5.35 MB` | `11.42 ms` | `11.10 ms` | `13.45 ms` | `15.20 ms` | `87.6 FPS` |
| **ONNX Runtime (`CUDAExecutionProvider`)** | `10.24 MB` | `6.18 ms` | `5.95 ms` | `7.12 ms` | `8.04 ms` | **`161.8 FPS`** |
| **Full GNC Loop (ONNX + 3D DEM Slope + $L_2$ Dist)** | `10.24 MB` | `8.45 ms` | `8.20 ms` | `9.65 ms` | `10.80 ms` | **`118.3 FPS`** |

> **Flight Computer Readiness:** With a **P99 tail latency of `10.80 ms`** for the complete perception-plus-planning loop, the system runs at **>90 Hz worst-case**, comfortably exceeding standard $10\text{–}20\text{ Hz}$ ($50\text{–}100\text{ ms}$) powered-descent GNC guidance update requirements.

---

## ⚠️ Domain Gap Analysis & Known Physical Boundary Conditions

1. **Why Zero-Shot Sim-to-Real Transfer Works:** Rendering training frames via Lambertian normal shading ($\mathbf{n} \cdot \mathbf{s}$) from continuous 3D DEMs and applying **CLAHE contrast normalization** (`clipLimit=2.0`, $8 \times 8$ grid) aligns synthetic histogram distributions with real LROC Narrow Angle Camera (NAC) imagery, allowing the network to key on directional shadow-to-lit-rim dipoles.
2. **Sub-Resolution Boulder Aprons ($<2\text{ m}$ diameter):** Ejecta blocks smaller than $4 \times 4\text{ px}$ fall below the stride-8 feature pyramid limit of `YOLO11n`. However, the **3D DEM slope gate ($\theta_{\text{slope}} \le 10^\circ$)** acts as a deterministic fallback: rough boulder fields and steep rille walls produce sharp elevation gradients that are masked out of `safe_mask` even when individual rocks are not boxed by YOLO.
3. **Extreme Polar Grazing Illumination ($<5^\circ$ Solar Elevation):** Near lunar south-pole permanently shadowed regions (e.g., Shackleton rim), cast umbra shadows stretch across multiple crater diameters and merge into contiguous black regions, occasionally causing adjacent craters to merge into a single bounding box.

---

## ⚡ Quick Start & Edge Deployment

### 1. Run Complete Training, Validation & Export Pipeline
```bash
pip install ultralytics opencv-python-headless matplotlib pyyaml onnx onnxruntime
python lunar_hazard_avoidance.py
```
Running `lunar_hazard_avoidance.py` will automatically:
* Synthesize the coupled 3D DEM + Optical training (`train`), in-distribution (`val_id`), and out-of-distribution (`val_ood`) datasets.
* Train `YOLO11n` for 20 epochs with mosaic and scale augmentations.
* Run zero-shot inference and 3D slope estimation on 6 real NASA lunar regions.
* Export `best.onnx`, run the P50/P95/P99 latency benchmark, generate `README_BENCHMARK.md`, and package all artifacts into `Lunar_HDA_Complete_Submission.zip`.

### 2. Compile to TensorRT Engine (For NVIDIA Jetson Orin / Xavier)
To deploy the exported ONNX model onto an embedded NVIDIA Jetson flight computer with FP16 hardware acceleration:
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/lunar_yolo11n_best.onnx \
    --saveEngine=models/lunar_yolo11n_fp16.engine \
    --fp16 \
    --workspace=1024
```

---

## 🛠️ Tech Stack

* **Deep Learning & Edge Inference:** PyTorch, Ultralytics YOLO11 (`yolo11n`), ONNX, ONNX Runtime (`CUDAExecutionProvider` / `CPUExecutionProvider`)
* **3D Terrain Physics & GNC Mathematics:** OpenCV (`cv2.distanceTransform`, `cv2.Sobel`, `cv2.createCLAHE`), NumPy (`np.fft` Frankot-Chellappa Poisson Photoclinometry)
* **Visualization & Reporting:** Matplotlib, PyYAML

---

## 👨‍💻 Author & Maintainer

**ASHISH TOMAR**  
*AI/ML Engineer & Aerospace Systems Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ashish%20Tomar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/ashish-tomar-/)
[![GitHub](https://img.shields.io/badge/GitHub-Tomar--Ashish--Singh-181717?style=flat&logo=github)](https://github.com/Tomar-Ashish-Singh/)
