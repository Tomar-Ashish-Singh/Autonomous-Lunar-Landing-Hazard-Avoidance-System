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

## 🏛️ Repository Architecture

The repository is structured following production MLOps standards, separating data pipelines, training artifacts, notebook environments, and inference scripts:

```text
Lunar-Landing-Hazard-Avoidance/
│
├── dataset/                        # Structured YOLO-format dataset directory
│   ├── images/
│   │   ├── train/                  # Procedurally generated training images (Regolith & Craters)
│   │   └── val/                    # Validation images for telemetry testing
│   └── labels/
│       ├── train/                  # YOLO normalized bounding box text files (class x_center y_center w h)
│       └── val/                    # Validation label text files
│
├── notebooks/
│   └── lunar_hazard_avoidance.ipynb # Complete, self-contained Google Colab training notebook
│
├── runs/
│   └── detect/
│       └── Lunar_Lander/           # Training runs, validation confusion matrices, and weight outputs
│           └── yolo_weights/
│               └── weights/
│                   └── best.pt     # Production-trained neural checkpoint
│
├── weights/
│   └── best.pt                     # Exported production weights for edge inference deployment
│
├── .gitignore                      # Standard Python and Ultralytics ignore rules
└── README.md                       # Comprehensive project documentation (This file)
