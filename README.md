# 🌱 ML-Powered Crop Recommendation System

This project is a **Machine Learning-powered Crop Recommendation System** built with **Streamlit**. It takes **soil and climate conditions** as input and recommends the most suitable crop for cultivation. The model is trained on agricultural datasets and uses feature scaling to ensure accurate predictions.

---

## 🚀 Features

* Interactive **Streamlit web app**
* Inputs for key soil & climate features:

  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * pH value
  * Temperature (°C)
  * Humidity (%)
  * Rainfall (mm)
* ML model prediction of the **best crop** for given conditions
* Automatic display of **crop images** from curated image links
* Clean, user-friendly interface

---

## 📂 Project Structure

```
├── streamlit_app.py             # Main Streamlit application
├── Crop_recommendation_system.pkl  # Trained ML model (saved with joblib)
├── scaler.pkl                   # Feature scaler for preprocessing
├── features_names.pkl           # Feature names used for the model
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run streamlit_app.py
```

---

## 📦 Dependencies

The main dependencies include:

* `streamlit` – for the web app UI
* `numpy` – numerical computing
* `pandas` – data handling
* `scikit-learn` – ML model and preprocessing
* `joblib` – saving & loading model artifacts

Install all dependencies via `requirements.txt`.

---

## 🧠 How It Works

1. User provides soil & climate inputs via the UI.
2. Inputs are converted into a **feature vector** with the correct column names.
3. Features are scaled using the pre-trained scaler.
4. The ML model predicts the best crop.
5. The system displays the recommended crop **with an image**.

---

## 🌾 Example Prediction

**Input:**

* N = 90
* P = 42
* K = 43
* Temperature = 26°C
* Humidity = 80%
* pH = 6.5
* Rainfall = 200 mm

**Output:**
✅ Recommended Crop: **RICE**


---

## 📸 Crop Images

The app includes curated image links for crops such as rice, maize, chickpea, kidney beans, pigeon peas, lentil, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee, and more.

---

## 🔮 Future Improvements

* Expand dataset for better generalization
* Add crop yield prediction (not just recommendation)
* Multi-language support for farmers
* Offline desktop/mobile app version

---

## 🙌 Acknowledgments

* Dataset sourced from publicly available crop recommendation datasets
* Built with [Streamlit](https://streamlit.io/)
* Images sourced from [Unsplash](https://unsplash.com) and other free image providers
