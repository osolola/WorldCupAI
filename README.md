# ⚽ FIFA World Cup AI Predictor

An end-to-end Machine Learning application that predicts the outcomes of international football matches using a hybrid **XGBoost** and **Elo Rating** engine.

## 🚀 Features

* **Dynamic Elo System:** Calculates real-time team strength ratings based on historical match performance.
* **Machine Learning Model:** Uses an XGBoost Classifier to predict Win/Draw/Loss probabilities.
* **Interactive Simulation:**
    * **Head-to-Head:** Detailed forecast for any hypothetical matchup.
    * **2022 Replay:** Re-run the 2022 World Cup to see how the AI's predictions differ from reality.
    * **Custom Bracket:** Build your own "What-If" tournament scenarios (e.g., Round of 16).

## Tech Stack

* **Python 3.9+**
* **Streamlit** (Web Interface)
* **XGBoost** (Gradient Boosting Classification)
* **Pandas & NumPy** (Data Manipulation)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/world-cup-ai.git](https://github.com/YOUR_USERNAME/world-cup-ai.git)
    cd world-cup-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run world_cup_prediction.py
    ```