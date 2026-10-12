# Autonomous Lunar Landing HDA — Benchmark & Transferability Report

## 1. Synthetic Generalization (In-Distribution vs. OOD Terminator Stress Test)

| Evaluation Split | mAP@50 | mAP@50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **In-Distribution (`val_id`)** | 0.049 | 0.014 | 0.016 | 0.659 |
| **Out-of-Distribution (`val_ood`)** | 0.037 | 0.008 | 0.025 | 0.582 |

## 2. Zero-Shot Transfer on Real NASA Lunar Imagery (With 3D DEM Slope Gating)

| Real Lunar Region | Detected Craters | Mean Conf | Safe Radius (m) | Touchdown Slope (deg) | 98th Pct Scene Slope (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LRO-01: Daedalus Central Peak & Floor (Offline Sim) | 0 | 0.000 | 0.0 m | 83.19 deg | 85.9 deg |
| LRO-02: Highland Degraded Crater Rim (Offline Sim) | 0 | 0.000 | 0.0 m | 84.72 deg | 85.7 deg |
| LRO-03: Mare-Highland Contact Zone (Offline Sim) | 0 | 0.000 | 0.0 m | 82.90 deg | 84.5 deg |
| LRO-04: Dense Secondary Crater Cluster (Offline Sim) | 0 | 0.000 | 0.0 m | 79.50 deg | 84.8 deg |
| LRO-05: Grazing Terminator Shadow Field (Offline Sim) | 0 | 0.000 | 0.0 m | 80.10 deg | 84.4 deg |
| LRO-06: Apollo Nadir Approach Corridor (Offline Sim) | 0 | 0.000 | 0.0 m | 69.06 deg | 84.4 deg |

## 3. Edge Deployment Benchmark (PyTorch vs. ONNX Runtime)

| Runtime / Pipeline Stage | Weight Size (MB) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Eager (best.pt)** | 5.23 MB | 11.41 | 10.91 | 15.61 | 16.75 | **87.6 FPS** |
| **ONNX Runtime (CPUExecutionProvider)** | 10.11 MB | 63.94 | 59.20 | 88.70 | 100.76 | **15.6 FPS** |
| **Full GNC Loop (ONNX + 3D DEM + DistTransform)** | 10.11 MB | 78.25 | 72.75 | 110.95 | 122.18 | **12.8 FPS** |