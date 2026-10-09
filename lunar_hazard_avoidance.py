
# AUTONOMOUS LUNAR LANDING HAZARD AVOIDANCE (SYNTHETIC MLOPS EDITION)

!pip install -q ultralytics opencv-python-headless matplotlib

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files


# PRE-FLIGHT: CLEANUP

print("🧹 Cleaning environment...")
os.system('rm -rf dataset runs')
for d in ['dataset/images/train', 'dataset/images/val', 'dataset/labels/train', 'dataset/labels/val']:
    os.makedirs(d, exist_ok=True)


# PHASE 1: PROCEDURAL SYNTHETIC LUNAR DATASET GENERATION

print("🚀 PHASE 1: Generating Procedural Lunar Dataset (Zero-API)...")

def create_yolo_dataset(num_images=150, img_size=512):
    for i in range(num_images):
        split = 'train' if i < int(num_images * 0.8) else 'val'
        
        # Base terrain: Grey surface with Gaussian noise for regolith texture
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 120
        noise = np.random.normal(0, 15, (img_size, img_size, 3)).astype(np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        num_craters = random.randint(3, 8)
        labels = []

        for _ in range(num_craters):
            r = random.randint(20, 60)
            cx, cy = random.randint(r, img_size-r), random.randint(r, img_size-r)
            
            # Draw crater (shadowed inside, highlighted rim for orbital sun-angle simulation)
            cv2.circle(img, (cx, cy), r, (80, 80, 80), -1) # Dark interior
            cv2.circle(img, (cx-int(r*0.2), cy-int(r*0.2)), int(r*0.8), (60, 60, 60), -1) # Deep shadow
            cv2.circle(img, (cx, cy), r, (160, 160, 160), 2) # Sunlit Rim
            
            # YOLO format: class x_center y_center width height (normalized)
            labels.append(f"0 {cx/img_size:.6f} {cy/img_size:.6f} {(r*2)/img_size:.6f} {(r*2)/img_size:.6f}")

        cv2.imwrite(f'dataset/images/{split}/lunar_{i}.jpg', img)
        with open(f'dataset/labels/{split}/lunar_{i}.txt', 'w') as f:
            f.write('\n'.join(labels))

create_yolo_dataset()
with open('dataset/dataset.yaml', 'w') as f:
    f.write("path: ../dataset\ntrain: images/train\nval: images/val\nnames:\n  0: crater")

print("✅ Synthetic Dataset successfully generated.")


# PHASE 2: NEURAL SCHEMA TRAINING (YOLO11)

print("\n🧠 PHASE 2: Initiating YOLO11 Training Sequence...")
model = YOLO('yolo11n.pt') 

results = model.train(
    data='dataset/dataset.yaml',
    epochs=15, 
    imgsz=512, 
    batch=16, 
    device=0, 
    project='runs/detect/Lunar_Lander',
    name='yolo_weights'
)

print("\n📊 --- TRAINING ACCURACY METRICS ---")
print(f"Mean Average Precision (mAP50): {results.box.map50:.3f}")
print(f"Precision: {results.box.p[0]:.3f}")
print(f"Recall: {results.box.r[0]:.3f}")
print("-------------------------------------\n")


# PHASE 3: PROBLEM SOLVING (Safe Zone Navigation)
print("🎯 PHASE 3: Autonomous Navigation & Inference Testing...")

import glob
val_images = glob.glob('dataset/images/val/*.jpg')
test_img = random.choice(val_images)
img = cv2.imread(test_img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# AI Inference
res = model.predict(img, conf=0.3, verbose=False)[0]
boxes = res.boxes.xyxy.cpu().numpy().astype(int)

# Distance Transform to find mathematically safest landing spot
hazard_mask = np.ones((512, 512), dtype=np.uint8) * 255
for (x1, y1, x2, y2) in boxes:
    pad = 15 # Spacecraft safety padding
    cv2.rectangle(hazard_mask, (max(0, x1-pad), max(0, y1-pad)), 
                  (min(512, x2+pad), min(512, y2+pad)), 0, -1)
                  
dist_transform = cv2.distanceTransform(hazard_mask, cv2.DIST_L2, 5)
_, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_rgb)
for (x1, y1, x2, y2) in boxes:
    ax[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', lw=2))
ax[0].set_title(f"Hazards Detected: {len(boxes)}")
ax[0].axis('off')

ax[1].imshow(img_rgb)
ax[1].add_patch(plt.Circle(max_loc, int(max_val), color='green', fill=True, alpha=0.4))
ax[1].plot(max_loc[0], max_loc[1], 'w+', markersize=15, markeredgewidth=3)
ax[1].set_title(f"Safest Landing Zone (R={int(max_val)}px)")
ax[1].axis('off')
plt.tight_layout()
plt.show()


# PHASE 4: EXPORT TO DEVICE

print("💾 PHASE 4: Exporting trained neural weights to your device...")
best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')

if os.path.exists(best_weights_path):
    files.download(best_weights_path)
    print("✅ Download triggered!")
else:
    fallback = glob.glob('runs/detect/**/weights/best.pt', recursive=True)
    if fallback:
        files.download(max(fallback, key=os.path.getctime))
        print("✅ Download triggered via fallback path.")
