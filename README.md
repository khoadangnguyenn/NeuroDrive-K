
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

---
## 🏗️ System Architecture

NeuroDrive-K operates on a modular, 5-layer pipeline:

1. **Sensing:** Multi-sensor data acquisition (Camera, LiDAR, Radar, GPS/IMU).
2. **Perception:** 3D object detection, segmentation, and unified Bird’s Eye View (BEV) mapping.
3. **Prediction:** Probabilistic trajectory forecasting and Bayesian risk assessment.
4. **Planning:** Multi-tiered decision making (Global Routing ➔ Tactical Behavior ➔ Local Trajectory).
5. **Control:** Precise vehicle actuation using PID and MPC controllers.

---
## 📂 Project Structure

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

## 🚀 Getting Started

```
# 1. Clone Repository
git clone https://github.com/your-username/neurodrive.git
cd neurodrive
```

```
# 2. Setup Environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

```
# 3. Run
python main.py --mode train
```

