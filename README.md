# 🌔 Autonomous Lunar Landing Hazard Avoidance System
*Real-time computer vision and spatial optimization pipeline for automated spacecraft descent and safe-zone selection.*

![Project Banner](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![YOLOv11](https://img.shields.io/badge/Model-YOLO11n-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview
During lunar or planetary descents, communication latency between Earth and the spacecraft (ranging from seconds to minutes) makes manual pilot intervention impossible. Landers must autonomously perceive surface hazards (craters, boulders, steep slopes) and instantly compute a safe touchdown vector.

This project implements an end-to-end **Space ML pipeline** that combines state-of-the-art object detection (**YOLO11**) with a spatial **Euclidean Distance Transform Algorithm** to dynamically identify, avoid, and target optimal safe landing zones in real time.

---

## 🧠 System Architecture & Workflow

The pipeline operates across four core phases:

1. **Synthetic Data Engineering / Synthesis:** Generates high-fidelity lunar regolith textures complete with orbital sun-angle shadow physics and multi-layered crater geometry.
2. **Neural Schema Training (YOLO11 Nano):** Fine-tuned on edge-optimized hardware constraints, achieving ultra-low inference latency suitable for flight computers.
3. **Hazard Masking & Clearance Buffering:** Converts detected crater bounding boxes into an occupancy grid with a configurable spacecraft safety margin (padding).
4. **Euclidean Distance Transform Optimization:** Calculates the mathematical gradient field across the terrain map to locate the absolute furthest coordinate from any hazard, designating it as the target landing center.
