
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
- 📡 **Multi-sensor data fusion** (Camera, LiDAR, Radar, GPS)
- 👁️ **Advanced perception** utilizing Bird’s Eye View (BEV)
- 🎲 **Probabilistic prediction** and robust risk assessment
- 🧠 **Hybrid planning** (Rule-based constraints + Learning-based algorithms)
- ⚙️ **Robust low-level vehicle control**


# 🚗 NeuroDrive-K: Hybrid Framework for Autonomous Driving systems

NeuroDrive is a modular, end-to-end autonomous driving framework designed to operate in complex urban environments. It bridges perception, prediction, planning, and control into a unified pipeline, combining Deep Learning with Hybrid AI to ensure both high performance and safety-critical reliability.

# 📌 Overview

NeuroDrive aims to simulate and eventually support real-world autonomous driving by integrating:

* Multi-sensor data fusion (Camera, LiDAR, Radar, GPS)
* Advanced perception using Bird’s Eye View (BEV)
* Probabilistic prediction and risk assessment
* Hybrid planning (rule-based + learning-based)
* Robust low-level vehicle control


# 🏗️ System Architecture

The system is divided into five core layers:

1. Sensing Layer

Collects raw environmental data:

* Camera (RGB/Stereo): Semantic understanding
* LiDAR: Precise 3D geometry
* Radar & GPS: Localization and motion sensing


2. Perception & BEV World Model

Transforms raw sensor data into structured understanding:

* Object detection, segmentation, lane detection
* BEV (Bird’s Eye View): Unified top-down spatial representation
* Occupancy Grid: Free vs. occupied space modeling


3. Prediction & Risk Assessment

Estimates future states of dynamic agents:

* Behavior prediction using probabilistic models
* Binary risk classification (e.g., collision likelihood)
* Bayesian Risk Modeling: Adaptive safety under uncertainty (e.g., weather)


4. Hybrid Planning

Decision-making system with three levels:

* Global Planning: Route planning (A*, Dijkstra)
* Behavior Planning: High-level decisions (stop, yield, overtake)
* Local Planning:
    * Cost Map Fusion (Risk + Rules + Distance)
    * Optimal trajectory generation via A*


5. Control Layer

Executes motion commands:

* Steering, acceleration, braking
* Controllers: PID / MPC
* Ensures smooth and stable vehicle behavior

```
📂 Project Structure (Suggested)

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

1. Clone Repository
```
git clone https://github.com/your-username/neurodrive.git
cd neurodrive
```
2. Setup Environment
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```
3. Run Simulation / Training
```
python main.py --mode train
```

# 📊 Evaluation

NeuroDrive supports evaluation using:

* Trajectory Metrics: AOE, ADE, MADE
* Safety Metrics: Collision Rate, Minimum Distance
* Scenario Testing: Seen vs Unseen environments


# 🔮 Future Work

* End-to-end learning integration
* Multi-agent interaction modeling
* Real-time deployment optimization
* Integration with CARLA / real-world datasets
