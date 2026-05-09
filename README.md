
# Benchmarking DINO for Drone-View Object Detection on VisDrone

> **CSE 429: Computer Vision and Pattern Recognition - Final Project**

<p align="center">
  <img src="./Deliverables/figures/teaser.png" alt="Teaser Image: YOLO vs DINO on VisDrone" width="100%">
</p>

## 📖 Abstract
The rapid proliferation of Unmanned Aerial Vehicles (UAVs) demands robust, high-precision aerial object detection. However, standard architectures struggle with extreme scale variations, high object density, and severe background clutter. In this project, we implement and benchmark **DINO (DETR with Improved DeNoising Anchor Boxes)** against the industry-standard **YOLOv11** and a vanilla **DETR** baseline on the challenging VisDrone-DET2019 dataset. 

Our findings validate the efficacy of Contrastive DeNoising (CDN) and anchor-free transformer logic in overcoming bipartite matching instability for small-scale targets, while revealing critical trade-offs in edge deployment viability.

---

## 🧠 Approach & Architecture

Standard detectors rely on dense anchor placement and Non-Maximum Suppression (NMS), which creates bottlenecks in crowded drone imagery. We bypass this using DINO, leveraging three core pillars:

1. **Contrastive DeNoising (CDN):** Injects positive and negative samples during training to stabilize bipartite matching and eliminate duplicate predictions without NMS.
2. **Mixed Query Selection:** Initializes positional queries dynamically from multi-scale encoder features rather than static priors, accelerating convergence on varied drone perspectives.
3. **Look Forward Twice:** A novel gradient refinement mechanism that allows deeper decoder layers to correct the bounding box coordinates of earlier layers, critical for localizing sub-15x15 pixel targets.

<p align="center">
  <img src="./Deliverables/figures/mixed query.png" alt="Architecture Flow" width="80%">

</p>

---

## 📊 Quantitative Results

Models were evaluated on the VisDrone validation split using an NVIDIA T4 GPU. 

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Speed (FPS) | Complexity (GFLOPs) |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv11 (Baseline)** | `0.304` | `0.173` | `28.7` | `195.5` |
| **Vanilla DETR** | `0.081` | `0.034` | `25.0` | `86.0` |
| **DINO (Proposed)** | `0.454` | `0.265` | `11.0` | `289.3` |

---

## 👁️ Qualitative Analysis

### Dense Urban Intersection

| YOLOv11x | DINO |
|---|---|
| ![](./Deliverables/figures/yolo_intersection.jpeg) | ![](./Deliverables/figures/dino_intersection.jpeg) |

**Figure:** Dense urban intersection. Left: YOLOv11x. Right: DINO. DINO detects substantially more pedestrians in the crowded upper regions, where NMS in YOLOv11x suppresses overlapping instances.

---

### Sparse Parking Lot

| YOLOv11x | DINO |
|---|---|
| ![](./Deliverables/figures/yolo_parking.jpeg) | ![](./Deliverables/figures/dino_parking.jpeg) |

**Figure:** Sparse parking lot. Left: YOLOv11x. Right: DINO. Both models detect the parked vehicles with high confidence; DINO additionally identifies pedestrians near the vehicles that YOLOv11x misses, but introduces some low-confidence false positives in shadow regions.

---

### Oblique Urban Street with Tree Canopy Occlusion

| YOLOv11x | DINO |
|---|---|
| ![](./Deliverables/figures/yolo_street.jpeg) | ![](./Deliverables/figures/dino_street.jpeg) |

**Figure:** Oblique urban street with heavy tree canopy occlusion. Left: YOLOv11x. Right: DINO. YOLOv11x detects vehicles confidently but misses most pedestrians occluded under the trees. DINO recovers a large number of pedestrian instances across the scene, including under canopy, at the cost of label density.

DINO excels in hyper-dense pedestrian plazas, successfully mapping distinct queries to heavily overlapping targets where YOLO's NMS aggressively merges them. Conversely, DINO occasionally lacks the localized inductive biases of CNNs, leading to false positives under low-light conditions by mistaking structural shadows for small vehicles.

---

## 🚀 Reproduction & Setup

### 1. Environment Installation
```bash
git clone https://github.com/tasbeeh04/DINO_DETR_UAV_Applications.git
cd DINO_DETR_UAV_Applications
```

### 2. Training & Inference

Due to extreme memory constraints of multi-scale transformers, we recommend executing our provided notebooks on Kaggle (minimum 16GB VRAM required).

* `notebooks/detr-with-visdrone.ipynb`
* `notebooks/DINO vs YOLO11x on COCO.ipynb`
* `notebooks/DINO vs YOLO11x on VisDrone.ipynb`
---

## 👥 Team

**Egypt-Japan University of Science and Technology (E-JUST)**

* Jana Mohamed Elsalhy
* Shiref Ashraf Mohamed
* Ahmed Nagah Ramadan
* Tasbih Othman Neamatallah
* Karim Yaser Mazroua
* Yasmeen Sameh Mohamed

**Instructor:** Prof. Ahmed Gomaa

---

## 📚 References

* [Vision Meets Drones: A Challenge (Zhu et al.)](https://arxiv.org/abs/1804.07437)
* [DINO: DETR with Improved DeNoising Anchor Boxes (Zhang et al.)](https://arxiv.org/abs/2203.03605)
