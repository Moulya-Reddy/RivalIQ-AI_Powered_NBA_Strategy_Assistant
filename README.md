# 🏀 RivalIQ – AI-Powered NBA Strategy Assistant

RivalIQ is an AI-powered basketball analytics platform that combines **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, and **Reinforcement Learning (Q-Learning)** to generate data-driven scouting reports and recommend clutch-time strategies for NBA matchups.

The application retrieves live team statistics, builds contextual scouting reports using RAG, evaluates the generated insights, and trains a reinforcement learning agent to optimize late-game decision making.

---

## ✨ Features

- 🏀 Live NBA team statistics
- 🤖 AI-generated scouting reports using RAG
- 📚 Retrieval-Augmented Generation pipeline
- 🧠 Reinforcement Learning (Q-Learning) strategy optimization
- 📈 Team comparison analytics and visualizations
- 📊 Interactive Streamlit dashboard
- ⚡ Groq-powered LLM inference
- 📋 Evaluation framework for RAG and RL performance

---

## 🏗️ System Architecture

```text
                    NBA Data API
                         │
                         ▼
              Data Collection & Analytics
                         │
        ┌────────────────┴───────────────┐
        ▼                                ▼
 Retrieval-Augmented Generation      Reinforcement Learning
        │                                │
        ▼                                ▼
 AI Scouting Report            Clutch Strategy Recommendation
        └──────────────┬─────────────────┘
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
│
├── data/
│
├── rag/
│
├── rl/
│
├── genai/
│
└── eval/
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API (Llama 3)
- Retrieval-Augmented Generation (RAG)
- Reinforcement Learning (Q-Learning)
- NumPy
- Matplotlib
- Requests
- BallDontLie NBA API
- python-dotenv

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/username/RivalIQ.git

cd RivalIQ
```

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate the environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```cmd
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
BALLDONTLIE_API_KEY=your_api_key
```

### Run the application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

## 📊 Evaluation

Run the evaluation suite

```bash
python eval/run_eval.py
```

The evaluation framework benchmarks:

- Retrieval quality
- RAG response quality
- Reinforcement learning performance
- Execution latency
- Token usage
- Estimated API cost

---

## 🎯 Design Decisions

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the language model, RivalIQ retrieves relevant basketball knowledge before generating scouting reports, improving factual accuracy and contextual relevance.

### Reinforcement Learning

A Q-Learning agent is trained to recommend clutch-time strategies based on game situations, enabling data-driven decision making.

### Modular Architecture

Each component (Data, RAG, RL, Evaluation) is implemented independently, making the system easy to extend and maintain.

### Interactive Dashboard

The Streamlit interface allows users to compare teams, generate scouting reports, and visualize model outputs in real time.

---

## 🔮 Future Improvements

- Multi-season analytics
- Player-level strategy recommendations
- Vector database integration
- Multi-agent collaboration
- Explainable AI dashboard
- Docker deployment
- Cloud deployment
- Historical matchup analysis

---

## 💡 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Reinforcement Learning
- Large Language Models (LLMs)
- Prompt Engineering
- AI Evaluation
- API Integration
- Streamlit
- Python Development
- Data Visualization
- Software Architecture

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Moulya Reddy Kandhala**

GitHub: https://github.com/Moulya-Reddy
