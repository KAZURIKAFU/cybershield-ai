# 🛡️ CyberShield AI — Network Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Dash](https://img.shields.io/badge/Dash-Plotly-purple?style=for-the-badge&logo=plotly)
![Cyber Security](https://img.shields.io/badge/Cyber_Security-NSDC_Certified-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Real-time ML-powered network intrusion detection with live threat dashboard**

> 🎓 Built by **Abhay Sharma** | Manipal University Jaipur  
> 🔐 Cyber Security Certified — NSDC / Launched Global × Nimesa

</div>

---

## 🚀 What is CyberShield AI?

CyberShield AI is a **real-time network intrusion detection system** that uses three machine learning models simultaneously to classify network traffic and detect cyber attacks as they happen.

---

## 🤖 ML Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 🌲 Random Forest | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| ⚡ SVM | 98.67% | 98.71% | 98.67% | 98.59% |
| 🧠 Neural Network | 98.17% | 97.99% | 98.17% | 97.99% |

---

## ⚔️ Attack Types Detected

| Attack | Severity | Description |
|---|---|---|
| 🟢 Normal | None | Regular legitimate traffic |
| 🟡 Port Scan | Low | Systematic port scanning |
| 🟠 Brute Force | Medium | Password cracking attempts |
| 🔴 SQL Injection | High | Database attack attempts |
| 🔴 DoS | High | Denial of Service flood |
| 🚨 DDoS | Critical | Distributed DoS attack |

---

## ✨ Dashboard Features

- 📡 **Live Traffic Monitor** — Real-time packet stream visualization
- 🤖 **3 ML Models** — Random Forest, SVM, Neural Network compared side by side
- 🚨 **Live Alert System** — Instant threat alerts with severity levels
- 📋 **Packet Log** — Real-time table of all network packets
- 📈 **24-Hour Timeline** — Historical attack frequency heatmap
- 🌍 **Top Attack Sources** — Countries & IPs behind attacks
- 🔒 **Security Score** — 0–100 network health score with letter grade
- 🔌 **Protocol Mix** — Traffic breakdown by protocol

---

## 🗂️ Project Structure

```
cybershield-ai/
├── app.py                      # Main Dash dashboard
├── ml_models.py                # Random Forest, SVM, Neural Network
├── traffic_simulator.py        # NSL-KDD style traffic generator
├── threat_analyzer.py          # Threat scoring & alert engine
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── LICENSE.txt                 # MIT License
└── assets/
    └── attack_signatures.json  # Attack pattern definitions
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/KAZURIKAFU/cybershield-ai.git
cd cybershield-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (models train automatically on startup ~30 seconds)
python app.py

# 4. Open in browser
# Navigate to: http://localhost:8052
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | Scikit-learn (Random Forest, SVM, MLP) |
| Data | NSL-KDD style simulated network traffic |
| Frontend | Dash by Plotly |
| Visualization | Plotly Express & Graph Objects |
| Real-time | Dash Interval callbacks (2.5s refresh) |

---

## 🔐 Certifications Demonstrated

- ✅ Cyber Security Mentorship — Launched Global × NSDC / Skill India
- ✅ Cyber Security Mentorship — Launched Global × Nimesa (Aug–Sep 2025)

---

## 👤 Author

**Abhay Sharma**  
🎓 B.Tech Data Science & Engineering — Manipal University Jaipur  
🔗 [LinkedIn](https://linkedin.com/in/abhay-sharma-426702208) | [GitHub](https://github.com/KAZURIKAFU)  
📧 abby.official2412@gmail.com

---

## 📄 License

MIT License — see [LICENSE.txt](LICENSE.txt) for details.

---

<div align="center">
⭐ Star this repo if you found it useful!  
Built with ❤️ by Abhay Sharma | Manipal University Jaipur
</div>
