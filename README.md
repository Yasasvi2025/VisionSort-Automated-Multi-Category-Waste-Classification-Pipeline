```markdown
# ♻️ VisionSort: Automated Multi-Category Waste Classification Pipeline

• VisionSort is a high-performance, deep learning-powered multi-stream waste segregation system. 
• Utilizing OpenAI's **CLIP (Contrastive Language-Image Pre-Training)** multi-modal model, the system executes **zero-shot classification** to instantly sort disposal materials into 10 distinct structural categories. 
• The architecture is engineered with a high-speed **FastAPI** backend and an intuitive, dark-themed industrial UI control center dashboard.

---

## 🎯 Key Features
* **Zero-Shot AI Classification:** Eliminates the need for traditional, intensive model training datasets by matching visual embeddings directly with a 10-class semantic text prompt matrix.
* **10-Class Target Matrix:** Covers Plastic, Metal, Glass, Paper, Cardboard, Organic, Medical Biohazards, E-Waste, Textiles, and Mixed Contaminated Waste.
* **Unified Single-File Architecture:** Seamlessly bundles a high-performance REST API backend with a responsive HTML5/JavaScript dashboard.
* **Industrial Monitoring Alerts:** Includes automated fault configuration protocols that flags low-confidence metrics (< 30%) for manual inspection conveyor protection.

---

## 🏗️ System Architecture & Workflow

The pipeline decouples ingestion, deep learning inference, and process monitoring into a streamlined, low-latency operational structure:

```text
=================================================================================================================
[1. DATA INGESTION TIER]             ►  [2. INFERENCE ENGINE TIER]        ►  [3. INDUSTRIAL MONITORING]
=================================================================================================================
• File payload raw bytes stream      • Tokenized text label matching      • Confidence check threshold (< 30%)
• RGB Conversion via Pillow          • CLIP Multi-modal computation       • Standard: STREAM VERIFIED
• Non-blocking FastAPI router        • Normalization via Softmax          • Fallback: PROCESS DISPATCH ALERT

```

---

## 📊 Industrial Engineering Competencies & Methodology

The development of the VisionSort framework incorporates core principles of automated process optimization, software engineering modularity, and operational risk mitigation:

### 1. Robust Zero-Shot Statistical Modeling

Traditional visual classifiers depend heavily on fixed output layers that collapse when processing unmapped materials. By shifting to a Multi-Modal Zero-Shot paradigm via OpenAI's CLIP, the pipeline establishes a highly adaptable semantic boundary. This ensures immediate system scaling to novel stream contaminants without stopping production for dataset collection or model retraining.

### 2. Low-Latency Microservice Architecture

* **State Decoupling:** By bundling the asynchronous routing capabilities of FastAPI with client-side asset rendering, the architecture completely avoids server-side blocking lag.
* **Efficient Memory Management:** The inference execution loop is explicitly enclosed in a `torch.no_grad()` context manager, shutting down the autograd engine during operations to minimize memory leaks and maximize conveyor processing speed.

### 3. Automated Quality Control & Fault Tolerance

Following strict industrial automation paradigms, the engine operates on a quantitative risk monitoring loop:

* **The Problem:** Poor lighting, camera sensor dust, or mangled packaging can cause weak classification metrics.
* **The Solution:** The system actively monitors operational variance. If incoming data signals drop below the $30\%$ threshold, the dashboard instantly shifts states to isolate the anomaly, prompting operators with a `⚠️ PROCESS DISPATCH NOTE` notice to audit the current stream lane.

---

## 🔬 Core Architecture Matrix Mapping

The deep learning layer evaluates target sample frames against a predefined metadata structure array:

| Class Index | Main Category | Display Structural Name |
| --- | --- | --- |
| **0** | Plastic | Polymer Recyclables / PET / HDPE |
| **1** | Metal | Ferrous & Non-Ferrous Scrap Metal |
| **2** | Glass | Glass Cullet / Intact Containers |
| **3** | Paper | High-Grade Paper / Office Waste |
| **4** | Cardboard | Corrugated Fibres / Box Packaging |
| **5** | Organic | Biological Food & Garden Waste |
| **6** | Medical | Clinical Biohazard / Syringes |
| **7** | E-Waste | Electronic Components & Circuitry |
| **8** | Textile | Fabric / Post-Consumer Garments |
| **9** | Mixed Waste | Residual Contaminated Refuse |

---

## 📂 Project Structure

```text
MajorProject/
├── Scripts/
│   └── main.py              # Main Application (FastAPI Core Engine + HTML Dashboard)
├── requirements.txt         # System Dependencies (PyTorch, FastAPI, Transformers, Pillow)
└── README.md                # Project Documentation & Architecture Blueprint

```

---

## 🛠️ Requirements & Environment Setup

Configure your core python workspace environment using the tracking packages inside your `requirements.txt`:

```text
fastapi
uvicorn
torch
transformers
pillow
python-multipart
numpy
jinja2
gradio

```

---

## 💻 Direct Environment Installation Command

Deploy the core dependencies with this single clean command:

```bash
pip install fastapi uvicorn torch transformers pillow python-multipart numpy jinja2 gradio

```

---

## 🚀 Local Installation & Setup Instructions

### 1. Clone the Codebase Repository

```bash
git clone https://github.com/Yasasvi2025/VisionSort-Automated-Multi-Category-Waste-Classification-Pipeline.git

```

### 2. Execution & Operational Workflow (Terminal Run)

Open your PowerShell terminal and run these exact directory navigation sequences to activate your environment and execute the server:

```powershell
# Navigate to your workspace directory drive
d:                                    
cd Pandu                              

# Activate the dedicated project virtual environment
MajorProject\Scripts\activate

# Navigate into the core executable script folder
cd MajorProject/Scripts               

# Launch the unified pipeline engine
python main.py                        

```

### Expected Startup Output Logs:

```text
Loading Semantic Vision Model (CLIP)...
VisionSort engine successfully online.
INFO:Started server process [127.0.0.1:6400]

```

* **Automated Launch:** A secure local browser window will automatically launch directly to your control console interface at: `http://127.0.0.1:6400/`

---

## 🌍 Cloud Distribution Deployment (Public Production Audit)

For live assessments without running a local Python setup, the system provides a live deployment link on the cloud container network:

* **Production Live Cloud Interface:** [VisionSort Production Control Center](https://huggingface.co/spaces/Yasasvi26/visionsort)

```

```
