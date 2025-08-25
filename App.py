# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# ------------------------------
# Load model, scaler, encoder, and feature names
# ------------------------------
model = joblib.load("Crop_recommendation_system.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features_names.pkl")  # list of feature names

# ------------------------------
# Streamlit UI
# ------------------------------
crop = 'orange'
st.set_page_config(page_title="🌱 Crop Recommendation System", layout="centered")

st.title("ML-Powered Crop Recommendation System🌾")
st.markdown("Provide soil and climate conditions, and I’ll recommend the most suitable crop for cultivation.")

# Input features
col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen(N) ratio", min_value=0.0, step=1.0)
    phosphorus = st.number_input("Phosphorus (P) ratio", min_value=0.0, step=1.0)
    potassium = st.number_input("Potassium (K) ratio", min_value=0.0, step=1.0)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.1)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, value=20.0, max_value=60.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
crop_images={
    'coffee':'https://plus.unsplash.com/premium_photo-1725551070322-e2900c896111?q=80&w=1283&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'rice':'https://images.unsplash.com/photo-1586201375761-83865001e31c?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'maize':'https://images.unsplash.com/photo-1615485290161-7eb49a34eba5?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'chickpea':'https://plus.unsplash.com/premium_photo-1675237624857-7d995e29897d?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'kidneybeans':'https://plus.unsplash.com/premium_photo-1671130295242-582789bd9861?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'pigeonpeas': 'https://as1.ftcdn.net/v2/jpg/05/77/57/70/1000_F_577577063_alcgd6wizneXmFHpufqWU4NiltfzXnKN.jpg',
    'mothbeans':'https://www.shutterstock.com/shutterstock/photos/2266072185/display_1500/stock-photo-moth-bean-turkish-gram-dew-bean-moth-dal-mat-matki-2266072185.jpg',
    'mungbean':'https://images.unsplash.com/photo-1594900799266-0e56587ba586?q=80&w=1001&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'blackgram':'https://imgs.search.brave.com/6KNl8PoYniiK0UT4410kbFJVcOk2arU75-yRUic2Hqw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy82/LzZmL0JsYWNrX2dy/YW0uanBn',
    'lentil':'https://plus.unsplash.com/premium_photo-1671130295987-13d3b3b4e9dc?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'pomegranate':'https://unsplash.com/photos/pomegranate-fruit-YjLywIe8vxE',
    'banana':'https://imgs.search.brave.com/YfNVDTkv0mWrLha45Z6q_ERHDFq3xRsDU7ioygocu5I/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzLzY3L2Zm/LzA0LzY3ZmYwNDMx/ZWQ0ZWNiZjEwZWJl/ZDkwYzE1ZWI2ZDBh/LmpwZw',
    'mango':'https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'grapes':'https://imgs.search.brave.com/j8N048Kd6ivpeWXvn_VH6Vwb2d_AS1JIBZTs75eUhcI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvMTc1/NDI4MDM1L3Bob3Rv/L3doaXRlLWFuZC1i/bGFjay1ncmFwZXMu/anBnP3M9NjEyeDYx/MiZ3PTAmaz0yMCZj/PXNiSk1oNVY3dnBQ/MlZKSnBEeFhZTGdK/RldrRFlhN3FRUU1v/b2ctSS1Fd1k9',
    'watermelon':'https://imgs.search.brave.com/DUQL7dvIkrGDG78umagxvdyVeisatqy9ndSJQjPraJI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzLzA2LzU5/L2MxLzA2NTljMTZl/MDJhYmNkOTQ1OTY4/NTVjMGI1YjJiNjdj/LmpwZw',
    'muskmelon':'https://imgs.search.brave.com/JeDs-LWJbpKgwNJ2rWxWqI2TEa4yzELxJv4_nHZxpD0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9tdXNr/bWVsb24tbXVza21l/bG9uLWlzb2xhdGVk/LXdoaXRlLWJhY2tn/cm91bmQtMTIzNzAx/MjU3LmpwZw',
    'apple':'https://images.unsplash.com/photo-1619546813926-a78fa6372cd2?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'orange':'https://images.unsplash.com/photo-1547514701-42782101795e?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'papaya':'https://imgs.search.brave.com/VG6E_Ze9F0_jn4Km8HavMOw6HMtxGS16qgFCdDcxEC4/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/cHJlbWl1bS1waG90/by9yaXBlLXBhcGF5/YS1pc29sYXRlZC13/aGl0ZS1jbGlwcGlu/Zy1wYXRoXzI2NjI4/LTkzNi5qcGc_c2Vt/dD1haXNfaHlicmlk/Jnc9NzQwJnE9ODA',
    'coconut':'https://media.istockphoto.com/id/2032230628/photo/coconut-isolated-coconuts-with-leaves-on-white-background-coconut-coco-half-and-leaf-full.jpg?s=2048x2048&w=is&k=20&c=-Qby2l6Ov4htfl0wBpvpjG9kGZq8ZyOkniLaj2h-uJs=',
    'cotton':'https://images.unsplash.com/photo-1616431101491-554c0932ea40?q=80&w=735&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'jute':'https://plus.unsplash.com/premium_photo-1674624789813-aee3aaa976cb?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
}
# ------------------------------
# Prediction
# ------------------------------
if st.button("🌿 Recommend Crop"):
    # Create input array
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

    # Convert into DataFrame with same feature names
    features_df = pd.DataFrame(features, columns=feature_names)

    # Scale the features
    features_scaled = scaler.transform(features_df)

    # Make prediction
    prediction = model.predict(features_scaled)
    crop = prediction[0]

    st.success(f"✅ The recommended crop for these conditions is: **{crop.upper()}**")
    if crop.lower() in crop_images:
        # st.image might not directly display search pages—download or grab raw image link for production
        st.image(crop_images[crop],use_container_width=True)
    else:
        st.write(f"No image available for: {crop}")