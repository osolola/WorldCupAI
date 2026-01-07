# ⚽ FIFA World Cup AI Predictor

[![Streamlit App] https://worldcupai.streamlit.app/ 

An end-to-end Machine Learning application that predicts the outcomes of international football matches using a hybrid **XGBoost** and **Elo Rating** engine.

## Features

* **Dynamic Elo System:** Calculates real-time team strength ratings based on historical match performance (1872–Present).
* **Machine Learning Model:** Uses an XGBoost Classifier to predict Win/Draw/Loss probabilities.
* **Interactive Simulation:**
    * **Head-to-Head:** Results for any hypothetical matchup.
    * **2022 Replay:** Re-run the 2022 World Cup to see how the AI's predictions differ from reality.
    * **Custom Bracket:** Build your own "What-If" tournament scenarios for Round of 16

## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit** (Web Interface)
* **XGBoost** (Gradient Boosting Classification)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Label Encoding)

## 📦 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/osolola/WorldCupAI.git](https://github.com/osolola/WorldCupAI.git)
    cd WorldCupAI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run world_cup_prediction.py
    ```

## 📊 Model Logic

The model uses a **diff-based approach**:
1.  **Elo Calculation:** Updates team ratings after every historical match.
2.  **Feature Engineering:** The model trains on `Home Elo`, `Away Elo`, and `Elo Difference`.
3.  **Prediction:** Outputs a probability distribution (e.g., Win 45% / Draw 30% / Loss 25%).

---
*Created by [osolola](https://github.com/osolola)*
