# IntelliVest: Intelligent Investment Profiling with Multi-Agent Systems

🚀 A cutting-edge agentic system built using the open-source **Agent Development Kit (ADK)** and powered by **Google Cloud Platform** to automate intelligent investor profiling, portfolio risk classification, and personalized investment insights.

---

## 🧠 Project Overview

**IntelliVest** demonstrates how autonomous agents can collaboratively solve complex financial problems—profiling potential investors, classifying their risk tolerance using real-world data, and delivering personalized wealth insights using scalable cloud technologies.

### 🔍 Problem Statement

Financial institutions need faster and more intelligent methods to assess client investment behavior and recommend risk-aligned portfolios. Traditional rule-based systems often fail to scale or personalize.

This project introduces a **multi-agent solution** using ADK, integrated with **BigQuery ML**, to automate:

- Investor profiling
- Risk classification (Conservative, Moderate, Aggressive)
- Personalized wealth recommendations

---

## 🎯 Features

- 🔁 **Multi-agent orchestration** using ADK (e.g., ProfilingAgent, RiskClassifierAgent, AdvisoryAgent)
- ☁️ **BigQuery ML** for training a classification model on investor behavior
- 🧪 Real-world dataset: UCI Bank Marketing with economic indicators
- ⚙️ Modular, production-grade code with cloud-native deployment using **Cloud Run**
- 📊 Live data exploration and model evaluation using **BigQuery UI**
- 📚 Complete setup scripts for ML model training and deployment

---

## 🏗️ Architecture

```mermaid
graph TD
  A[User Request] -->|via CLI/API| B[Agent Engine (ADK)]
  B --> C[Profiling Agent]
  C --> D[BigQuery ML Model]
  D --> E[RiskClassifier Agent]
  E --> F[Recommendation Agent]
  F -->|Results| G[Response (Web/UI/API)]
```

---

## 📁 Project Structure

```
intellivest/
│
├── agents/                    # ADK agents: Profiling, RiskClassifier, Advisory
│   ├── profiling_agent.py
│   ├── risk_classifier_agent.py
│   └── advisory_agent.py
│
├── workflows/                 # Multi-agent workflows
│   └── investment_flow.yaml
│
├── data/                      # Raw and processed datasets
│   └── bank-additional-full.csv
│
├── scripts/                   # One-time setup scripts
│   ├── create_bigquery_tables.py
│   └── train_bigquery_model.sql
│
├── notebooks/                 # Exploration and model performance
│   └── data_analysis.ipynb
│
├── .adk/                      # ADK configuration and metadata
│   └── config.yaml
│
├── Dockerfile                 # For deployment on Cloud Run
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/intellivest.git
cd intellivest
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up GCP & BigQuery

- Enable BigQuery, Cloud Run, and Artifact Registry
- Create a dataset and table using `scripts/create_bigquery_tables.py`
- Train the model using:

```bash
bq query --use_legacy_sql=false < scripts/train_bigquery_model.sql
```

### 4. Run locally

```bash
adk run workflows/investment_flow.yaml
```

### 5. Deploy to Cloud Run (Optional)

```bash
gcloud builds submit --tag gcr.io/<project-id>/intellivest
gcloud run deploy intellivest --image gcr.io/<project-id>/intellivest --platform managed
```

---

## 📊 Dataset

This project uses the **Bank Marketing Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

**Citation**:

> Moro, S., Cortez, P., & Rita, P. (2014). _A Data-Driven Approach to Predict the Success of Bank Telemarketing_. Decision Support Systems.
> [DOI: 10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)

Please cite the original authors if you use this dataset in your own work.

---

## 🚀 Google Cloud Usage

This project uses the following GCP services:

- **BigQuery ML**: For classification model training
- **Cloud Run**: For serverless deployment
- **Artifact Registry**: For container management
- **GCS (optional)**: For storing intermediate files

---

## 🧠 Agent Development Kit (ADK) Capabilities

- Declarative agent orchestration via YAML
- Reusable agent modules
- Pluggable LLMs (local or Vertex AI)
- Seamless tool integration (e.g., BigQuery client)

---

## 📽️ Demo (Coming Soon)

> Link to your demo video, screenshots or walkthrough

---

## 📘 License

MIT License

---

## 🙏 Acknowledgments

Thanks to the authors of the Bank Marketing dataset:
Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho), and Paulo Rita (ISCTE-IUL)

---
