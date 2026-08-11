# 🛡️ Sentinel-Sec: Integrated NIDS & Web Vulnerability Scanner

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green?logo=scikitlearn)](https://scikit-learn.org/)

**Sentinel-Sec** is an intelligent security dashboard providing multi-layered defense analysis. By integrating **Machine Learning** with **Live Packet Sniffing**, it identifies network-level intrusions and application-layer web attacks in a single unified interface.

---

## 🚀 Core Features

### 📡 1. Intruder Detector (Bulk Log Analysis)
* **Dataset:** Built on the **NSL-KDD (KDDTrain+)** dataset.
* **Accuracy:** Employs a **Random Forest Classifier** to achieve high-precision classification of "Normal" vs. "Attack" traffic.
* **Smart Training:** Features `st.cache_resource` auto-training logic that builds the model on-the-fly if `Model_Nids.pkl` is missing.
* **Visuals:** Provides traffic distribution charts and highlighted dataframes for immediate threat identification.

### ⚡ 2. Live Detection (Packet Sniffer)
* **DPI Engine:** Utilizes **Scapy** for Deep Packet Inspection of live TCP/80 traffic.
* **Injection Guard:** Intercepts raw HTTP payloads and uses a **TF-IDF Vectorizer** + **Random Forest** to detect SQLi and XSS attempts mid-transit.
* **Anomaly Detection:** Flags oversized packets (>1000 bytes) that may indicate data exfiltration or buffer overflow attempts.

### 🌐 3. Web Vulnerability Scanner
* **Form Auditor:** Automatically extracts forms and input fields from target URLs.
* **Attack Simulation:** Probes for **Reflected XSS** and **SQL Injection** vulnerabilities by analyzing server response patterns and database syntax error leaks.

### 💰 4. Financial Fraud & CDR Analyzer
* **Forensic Capability:** Specifically designed for analyzing Bank Logs, UPI Transactions, and Call Detail Records (CDR).
* **Intelligence:** Identifies repeat offenders, flags high-value transfers (₹50,000+), and offers a global search to cross-reference suspect IDs.

---

## 🛠️ Technical Stack

* **Frontend:** Streamlit
* **AI/ML:** Scikit-Learn, Pandas, NumPy, Joblib
* **Networking:** Scapy (Live Sniffing)
* **Web:** BeautifulSoup4, Requests (Vulnerability Scanning)

---

## 📦 Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/om-pakhale/Network-Intrusion-Detection-System-NIDS-.git](https://github.com/om-pakhale/Network-Intrusion-Detection-System-NIDS-.git)
   cd Network-Intrusion-Detection-System-NIDS-
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **External Requirements:**
   * **Windows:** Install [Npcap](https://nmap.org/npcap/) for live sniffing and scanning.
   * **Permissions:** Run the app with **Administrator** or **Sudo** privileges to access the network interface.

4. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Repository Structure
```text
├── app.py                     # Main Application Logic
├── KDDTrain+.txt              # Training Data (NSL-KDD)
├── full_injection_model.pkl   # AI Model for Web Injections
├── full_vectorizer.pkl        # TF-IDF Vectorizer for Payloads
├── Model_Nids.pkl             # Cached NIDS ML Model
└── requirements.txt           # Project Dependencies
```

---

## 👨‍💻 Developer
**Om Narendra Pakhale**
* **University:** DKTE Society's Textile & Engineering Institute
* **Branch:** B.Tech in Artificial Intelligence & Machine Learning
* **Profile:** [LinkedIn](https://www.linkedin.com/in/om-pakhale/) | [HackerOne](https://hackerone.com/om0405)

---

## ⚖️ Disclaimer
*Sentinel-Sec is intended for educational and authorized security testing purposes. Unauthorized use of this tool on networks or websites without permission is strictly prohibited.*


