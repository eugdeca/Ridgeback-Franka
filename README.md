# 🚀 Perceptive Mobile Manipulation: Asymmetric Reinforcement Learning for Franka-Ridgeback

![NVIDIA Isaac Sim](https://img.shields.io/badge/Simulation-NVIDIA%20Isaac%20Sim-76B900?logo=nvidia)
![Isaac Lab](https://img.shields.io/badge/Framework-Isaac%20Lab-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-PPO%20(Asymmetric)-red)
![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)

This repository presents an advanced **End-to-End Deep Reinforcement Learning** framework for a mobile manipulation task. We trained an 11-DoF robotic system (a 7-DoF Franka Emika Panda arm + gripper, mounted on a 3-DoF omnidirectional base) to autonomously reach and interact with a cube using **raw visual input** and proprioceptive data.

The project demonstrates how to overcome the high-dimensionality challenges of visual inputs in RL by implementing a state-of-the-art **Asymmetric Actor-Critic** architecture, paving the way for Sim-to-Real deployment.

---

## 🎥 Final Result in Action


https://github.com/user-attachments/assets/4b9f05a8-8dfc-40b5-8446-ca91c8eb0fce


> The Franka-Ridgeback agent successfully navigating and reaching the target using only depth vision and proprioceptive data.

---

## 🌟 Key Achievements & Selling Points

Based on extensive training and evaluation, our final policy achieved remarkable robustness and realism:

### 1. Robust Perceptive Policy via Asymmetric Training
Standard Multi-Layer Perceptrons (MLPs) often collapse when handling high-dimensional visual data. While CNN Distillation showed promise, it inherited sub-optimal behaviors from the Teacher. Our final **Asymmetric Actor-Critic approach** completely bypassed these issues. 
* **The Critic** leverages privileged information (exact state and cube poses) to guide the learning process.
* **The Actor** relies strictly on deployable sensory data (CNN-processed depth images + proprioception).
* **Result:** A highly robust policy that seamlessly adapts to dynamic environments and curriculum changes.

### 2. Sim-to-Real Readiness
A major challenge in RL is generating realistic motions. By integrating strict **Kinematic Constraints** into the reward shaping (penalizing high joint velocities), the trained policy completely avoids erratic and "jerky" movements. The resulting trajectories are smooth, realistic, and strictly adhere to the hardware limits of the real Franka robot, making the policy structurally ready for real-world transfer.

### 3. Advanced Curriculum Learning
To ensure stable convergence, the agent was trained using a highly optimized, step-wise **Curriculum Learning** strategy. The task difficulty scales progressively—from basic reaching to complex manipulation with visual dependencies—preventing early learning collapse and ensuring a high success rate.

---

## 🧠 System Architecture

* **Robot setup:** Franka Emika Panda (7-DoF Arm + Gripper) mounted on an Omnidirectional Base (3-DoF: Prismatic X, Prismatic Y, Revolute Z).
* **Environment:** Built on NVIDIA Isaac Lab.
* **Observation Space:** * *Actor:* Proprioceptive data (joint positions/velocities) + Depth Vision (processed via Convolutional Neural Network).
  * *Critic:* Full privileged state.
* **Algorithm:** Proximal Policy Optimization (PPO).

## 📦 Setup & Usage

To use this environment, simply extract the `Collected_ridgeback_franka.zip` archive directly inside the main task folder. This will correctly unpack and link all the required 3D USD assets and textures for the simulation.

## Authors
* **Eugenio Delli Carri** 
* **Lorenzo Incarnati** 
* **Artur Vagapov**
