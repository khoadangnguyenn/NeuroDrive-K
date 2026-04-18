
<div align="center">

# 🚗 NeuroDrive-K
**A Modular, Hybrid Framework for Autonomous Driving Systems**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![AI Framework](https://img.shields.io/badge/AI-Deep_Learning_%7C_Bayesian_Networks-ee4c2c.svg?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

*Bridging perception, prediction, planning, and control into a unified pipeline.*

</div>

<br>

## 📌 Overview

**NeuroDrive-K** is an end-to-end autonomous driving framework designed to operate in complex urban environments. By combining **Deep Learning** with **Hybrid AI**, it ensures both high-performance adaptability and safety-critical reliability. 

The framework aims to simulate and eventually support real-world autonomous driving by integrating:
- **Multi-sensor data fusion** (Camera, LiDAR, Radar, GPS)
- **Advanced perception** utilizing Bird’s Eye View (BEV)
- **Probabilistic prediction** and robust risk assessment
- **Hybrid planning** (Rule-based constraints + Learning-based algorithms)
- **Robust low-level vehicle control**

---

## 🏗️ System Architecture

**NeuroDrive-K** is built upon a modular, scalable pipeline. The system is compartmentalized into five core layers that collectively transform raw environmental data into actionable driving maneuvers:

### 1. 📡 Sensing Layer (Data Acquisition)
The foundation of the pipeline, responsible for real-time environmental data ingestion:
* **Camera (RGB/Stereo):** Extracts high-resolution visual context and semantic features.
* **LiDAR:** Captures precise 3D geometric point clouds for depth estimation and physical profiling.
* **Radar & GPS/IMU:** Provides robust localization, velocity tracking, and all-weather motion sensing.

### 2. 👁️ Perception & BEV World Model
Transforms heterogeneous sensor streams into a unified, structured understanding of the world:
* **Core Tasks:** 2D/3D object detection, semantic segmentation, and lane topology mapping.
* **Bird’s Eye View (BEV):** Fuses multi-modal data into a cohesive, top-down spatial representation.
* **Occupancy Grid Mapping:** Accurately delineates drivable free space from static and dynamic obstacles.

### 3. 🎲 Prediction & Risk Assessment
Anticipates the future states of dynamic agents to ensure proactive safety:
* **Trajectory Forecasting:** Utilizes probabilistic models to predict the intentions and future paths of surrounding traffic.
* **Risk Classification:** Real-time binary and multi-class collision likelihood estimation.
* **Bayesian Risk Modeling:** Dynamically adjusts safety margins under uncertainty (e.g., adverse weather, sensor noise, occlusion).

### 4. 🧠 Hybrid Planning
A robust, multi-tiered decision-making engine:
* **Global Planning:** High-level route formulation using graph search algorithms (A*, Dijkstra).
* **Behavioral Planning:** Tactical, context-aware decision-making (e.g., yielding, stopping, overtaking, lane changing).
* **Local Planning:** Generates spatio-temporally optimal, collision-free trajectories based on a fused Cost Map (evaluating risk, traffic rules, and obstacle clearance).

### 5. ⚙️ Control Layer
Translates planned trajectories into precise, low-level vehicle actuation:
* **Execution:** Commands for steering angle, acceleration (throttle), and braking.
* **Controllers:** Implements industry-standard algorithms (PID, Model Predictive Control - MPC).
* **Objective:** Ensures smooth, stable, and passenger-comfortable vehicle dynamics under diverse driving conditions.

---
# 📂 Project Structure

```

NeuroDrive/
│── sensing/              # Sensor data processing
│── perception/           # Detection, segmentation, BEV
│── prediction/           # Behavior prediction models
│── planning/
│   ├── global/
│   ├── behavior/
│   └── local/
│── control/              # PID / MPC controllers
│── configs/              # Config files
│── datasets/             # Training / evaluation data
│── utils/                # Common utilities
│── main.py               # Entry point
│── requirements.txt
│── README.md
```

# 🚀 Getting Started

## Clone Repository
```
git clone https://github.com/your-username/neurodrive.git
cd neurodrive
```
## Setup Environment
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```
## Run Simulation / Training
```
python main.py --mode train
```

