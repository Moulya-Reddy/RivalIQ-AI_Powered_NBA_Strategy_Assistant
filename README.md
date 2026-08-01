# 🏀 RivalIQ – AI-Powered NBA Strategy Assistant

RivalIQ is an AI-powered basketball analytics platform that combines **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, and **Reinforcement Learning (Q-Learning)** to generate data-driven scouting reports and recommend clutch-time strategies for NBA matchups.

The application retrieves live NBA statistics, builds contextual scouting reports using RAG, evaluates the generated insights through a faithfulness benchmark, and trains a reinforcement learning agent to recommend optimal late-game strategies.

---

## ✨ Features

- 🏀 Live NBA team statistics and analytics
- 🤖 AI-generated scouting reports using Retrieval-Augmented Generation (RAG)
- 📚 Retrieval pipeline for grounded report generation
- 🧠 Reinforcement Learning (Q-Learning) for clutch-time strategy optimization
- 📊 Interactive Streamlit dashboard
- 📈 Team comparison visualizations and performance analytics
- ⚡ Groq-powered LLM inference
- ✅ Faithfulness evaluation for generated reports
- 📋 Automated evaluation framework for RAG and RL components

---

## 📸 Demo

The **`docs/`** folder contains demonstration assets for the project:

- **`demo.mov`** – Walkthrough of the Streamlit application showing scouting report generation, analytics, and reinforcement learning workflow.
- **`run_eval.png`** – Screenshot of the evaluation framework displaying faithfulness scores and benchmark results.

These assets provide a quick overview of the project without requiring local setup.

---

## 🏗️ System Architecture

```text
                    NBA Data API
                         │
                         ▼
              Data Collection & Analytics
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Retrieval-Augmented Generation      Reinforcement Learning
        │                                 │
        ▼                                 ▼
 AI Scouting Report            Clutch Strategy Recommendation
        └────────────────┬────────────────┘
                         ▼
                 Streamlit Dashboard
```

---

## 📂 Project Structure

```text
RivalIQ/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── LICENSE
│
├── cache/
│
├── data/
│   ├── analytics.py
│   └── client.py
│
├── rag/
│   ├── generator.py
│   ├── notes.py
│   └── retriever.py
│
├── rl/
│   ├── agent.py
│   ├── environment.py
│   └── team_params.py
│
├── genai/
│   └── visuals.py
│
└── eval/
    ├── faithfulness.py
    ├── run_eval.py
    └── results.json
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API (Llama 3)
- Retrieval-Augmented Generation (RAG)
- Reinforcement Learning (Q-Learning)
- BallDontLie NBA API
- NumPy
- Matplotlib
- Requests
- python-dotenv

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Moulya-Reddy/RivalIQ.git

cd RivalIQ
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
BALLDONTLIE_API_KEY=your_balldontlie_api_key
```

### 6. Run the application

```bash
streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

---

## 📊 Evaluation

Run the evaluation framework using

```bash
python eval/run_eval.py
```

The evaluation measures:

- Retrieval quality
- Faithfulness of generated scouting reports
- Reinforcement learning performance
- Execution latency
- Token usage
- Estimated API cost

Example evaluation outputs are available in **`docs/run_eval.png`**.

---

## 📈 Sample Outputs

RivalIQ generates:

- AI-powered scouting reports grounded in retrieved NBA statistics
- Team comparison analytics
- Reinforcement learning strategy recommendations
- Faithfulness evaluation scores
- Performance metrics and visualizations

Refer to the **`docs/`** folder for example outputs.

---

## 🎯 Design Decisions

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the language model, RivalIQ retrieves relevant basketball knowledge before generating scouting reports. This improves factual grounding and reduces hallucinations.

### Reinforcement Learning

A Q-Learning agent is trained to recommend clutch-time strategies based on game situations, enabling data-driven tactical recommendations.

### Faithfulness Evaluation

Generated reports are evaluated against retrieved evidence to estimate how well the generated content is supported by the underlying data.

### Modular Architecture

Each component (Data Collection, RAG, Reinforcement Learning, Evaluation, and Visualization) is implemented independently, making the system easy to extend and maintain.

### Interactive Dashboard

The Streamlit interface allows users to compare teams, generate scouting reports, evaluate report quality, and visualize model outputs in real time.

---

## 🔮 Future Improvements

- Player-level matchup analysis
- Multi-season historical analytics
- Vector database integration
- Multi-agent scouting workflows
- Explainable AI dashboard
- Docker deployment
- Cloud deployment
- Advanced reinforcement learning algorithms
- Historical playoff matchup analysis

---

## 💡 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Reinforcement Learning (Q-Learning)
- Large Language Models (LLMs)
- Prompt Engineering
- AI Evaluation
- API Integration
- Streamlit
- Python Development
- Data Analytics
- Data Visualization
- Software Architecture

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Moulya Reddy Kandhala**

GitHub: https://github.com/Moulya-Reddy
