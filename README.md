
# 🌔 Autonomous Lunar Landing Hazard Avoidance System (Space ML)

An end-to-end Computer Vision and Spatial Optimization pipeline designed to solve real-time autonomous descent and hazard avoidance for planetary landers (Artemis / Chandrayaan missions). 

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![YOLOv11](https://img.shields.io/badge/Model-YOLO11n-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Executive Summary & Problem Statement

During planetary descent, communication round-trip latency between Earth and the spacecraft (ranging from seconds to minutes) makes manual ground control impossible. Landers cannot rely on pre-mapped global topography because local hazards (such as boulders, steep slopes, and secondary impact craters) are only resolved during terminal descent.

This project implements an autonomous edge-AI system that:
1. **Perceives** surface hazards in real-time from optical camera feeds using state-of-the-art object detection (**YOLOv11**).
2. **Computes** a spatial occupancy hazard mask with safety clearance padding.
3. **Optimizes** touchdown coordinates dynamically using a **Euclidean Distance Transform Algorithm** to find the absolute safest, flattest terrain farthest from all hazards.

---

## 🏛️ Simplified Repository Architecture

Here is the essential structure you need for your repository:

```text
Lunar-Landing-Hazard-Avoidance/

│└── lunar_hazard_avoidance.py # The main code
│
├── result.png                    
│
│
└── README.md                        # This documentation file

```

---

## 🛠️ Step-by-Step Setup Guide

Follow these steps to populate your GitHub repository properly:

### Step 1: Initialize the Repository

1. Go to GitHub and create a new repository named `Lunar-Landing-Hazard-Avoidance`.
2. Do **not** initialize it with a README, .gitignore, or license (we will add those manually).
3. Clone the empty repository to your local machine:
`git clone https://github.com/YourUsername/Lunar-Landing-Hazard-Avoidance.git`

### Step 2: Add the Core Code

1. Inside your cloned folder, create a directory named `notebooks`.
2. Download your finished Google Colab notebook (`.ipynb` file).
3. Rename it to `lunar_hazard_avoidance.ipynb` and place it inside the `notebooks` directory.

### Step 3: Add Result Images

1. Create a directory named `results` in the main folder.
2. Run your Colab notebook to generate the output plots.
3. Right-click and save the validation metrics plot as `validation_results.png`.
4. Right-click and save the final spatial optimization plot (showing the safe landing spot) as `landing_optimization.png`.
5. Place both images inside the `results` folder.

### Step 4: Add the README

1. Create a file named `README.md` in the main directory.
2. Copy the entire contents of this Markdown block and paste it into `README.md`.
3. *(Optional)* Update the image paths below in the "Results" section if you named your files differently.

### Step 5: Commit and Push

1. Open your terminal in the repository folder.
2. Run the following commands:

```bash
   git add .
   git commit -m "Initial commit: Added notebook, results, and README"
   git branch -M main
   git push -u origin main
   

```

---

## 🖼️ Visual Results

*(Ensure you have added your images to the `results/` folder for these to display correctly on GitHub)*




---

## 🔬 System Workflow & Architecture

The system pipeline bridges deep learning inference with spatial mathematics:

```text
[ Orbital Camera Feed ] 
         │
         ▼
[ YOLO11n Nano Backbone ] ──► (Detects Crater Bounding Boxes)
         │
         ▼
[ Hazard Occupancy Mask ] ──► (Binary Grid: 0 = Hazard, 255 = Free Terrain)
         │
         ▼
[ Euclidean Distance Transform ] ──► (Calculates Gradient Field of Distances)
         │
         ▼
[ Optimal Landing Coordinate ] ──► (Selects Global Maximum: Safest Touchdown Center)

```

### 1. Synthetic Data Engineering

Because real-time annotated lunar descent imagery is scarce, the pipeline features a built-in Procedural Data Synthesis Engine utilizing NumPy and OpenCV. It models:

* Regolith texture via multi-layered Gaussian noise distributions.
* Orbital sun-angle lighting effects by rendering dark crater interiors paired with offset sunlit rims.
* Automated bounding box label normalization matching strict YOLO formatting.

### 2. Neural Schema (YOLOv11 Nano)

* **Backbone:** CSPDarknet optimized for sub-10ms inference speeds on resource-constrained flight hardware.
* **Neck:** Path Aggregation Network (PANet) ensuring precise scaling for small, distant craters during high-altitude descent.
* **Loss Function:** Complete Intersection over Union (CIoU) loss combined with Distribution Focal Loss (DFL).


## 📊 Performance & Evaluation Metrics

Evaluated on the rigorous validation dataset after 15 epochs on an NVIDIA T4 GPU:

| Evaluation Metric | Score | Engineering Interpretation |
| --- | --- | --- |
| **Mean Average Precision ($mAP_{50}$)** | 0.965 | Exceptional overall detection accuracy across varying crater scales. |
| **Precision ($P$)** | 0.987 | Near-zero false-positive rate; prevents the spacecraft from aborting safe landings due to false alarms (e.g., shadows). |
| **Recall ($R$)** | 0.969 | High hazard-capture rate; minimizes the catastrophic risk of missing real obstacles. |
| **$mAP_{50-95}$** | 0.945 | Robust boundary localization across strict Intersection over Union thresholds. |
| **Inference Latency** | ~9.8 ms | Real-time flight computer execution capability. |

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch & Ultralytics YOLOv11
* **Computer Vision & Math:** OpenCV, NumPy, SciPy
* **Data Visualization:** Matplotlib
* **Execution Environment:** Google Colab (NVIDIA T4 GPU Runtime)

---




## 👨‍💻 Author & Maintainer

**ASHISH TOMAR**

*AI/ML Engineer & Aerospace Systems Enthusiast*

[LinkedIn](https://www.linkedin.com/in/ashish-tomar-/) | [GitHub](https://github.com/Tomar-Ashish-Singh/)


