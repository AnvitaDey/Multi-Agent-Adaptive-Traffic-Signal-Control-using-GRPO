# 🚦 Multi-Agent Traffic Signal Control using GRPO & PPO

## 📌 What this project is about

This project focuses on building an **adaptive traffic signal control system** using Reinforcement Learning (RL) in a simulated 3×3 urban road network.

It compares traditional methods like fixed-time control with modern RL approaches:

* PPO (multi-agent)
* GRPO (group-based multi-agent learning)

---

## Aim

The goal of this project is to show that:

> **GRPO-based multi-agent systems can learn coordinated traffic control policies and outperform traditional signal systems, especially under heavy traffic conditions.**

---

## Key Idea

* Each intersection is treated as an **agent**
* Agents learn from **local traffic conditions**
* Training is done on **mixed traffic densities** (light, moderate, heavy)
* GRPO introduces **group-relative learning** to improve coordination

---

## Core Result

* RL methods significantly reduce congestion under heavy traffic
* GRPO performs competitively with PPO and shows potential for better coordination

---

## Conclusion

This project demonstrates that **multi-agent RL, especially GRPO, is a promising approach for scalable and adaptive traffic signal control in complex urban environments.**

---
