# 🧠 Mental Health Disorder Risk Predictor

This is a Streamlit web app that uses Machine Learning (Random Forest Classifier) to predict whether an individual is likely to need mental health treatment based on personal and workplace-related factors.

## 🚀 Live Demo

🌐 [Click here to use the live app](https://my-username.streamlit.app)  
*(I will replace with my actual Streamlit Cloud link)*

---

## 📂 Project Structure


---

## 📊 Dataset

- **Source**: [Kaggle - Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)
- **Description**: Survey responses from tech employees about workplace mental health policies and personal experiences.

---

## 🛠️ Features

- Predicts likelihood of needing mental health treatment.
- Uses inputs like age, gender, remote work, family history, etc.
- Pre-trained with Random Forest and StandardScaler.
- Easy-to-use Streamlit interface.

---

## 🧪 How It Works

1. The user fills out a form with personal and workplace information.
2. The app encodes and scales the inputs using the saved `scaler.pkl`.
3. The trained model (`mental_health_predictor.pkl`) makes a prediction.
4. The app displays the result with recommendations.

---

## 📦 Installation (For Local Development)

Clone the repo and install dependencies:

```bash
git clone https://github.com/get-it-by-yourself-username/mental-health-predictor.git
cd mental-health-predictor
pip install -r requirements.txt
streamlit run mental_health_predictor_app.py

