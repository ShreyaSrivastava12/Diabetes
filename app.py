# app.py
import os
import streamlit as st
import joblib
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
# Load the trained Logistic Regression model
model = joblib.load(model_path)

# Function to predict diabetes using the model
def predict_diabetes(features):
    prediction = model.predict([features])[0]
    return prediction

# Streamlit app code
def main():
    st.title('Diabetes Detection')

    # Take user input for features
    pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
    glucose = st.slider('Glucose Level', 0, 400, 100)
    blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 300, 70)
    skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
    insulin = st.slider('Insulin Level (mu U/ml)', 0, 1000, 79)
    bmi = st.slider('BMI', 0.0, 80.0, 30.0)
    diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 3.00, 0.3725)
    age = st.slider('Age', 0, 120, 30)

    # Create a feature vector from user inputs
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

    # Get the prediction
    prediction = predict_diabetes(features)

    # Display the prediction
    if prediction == 0:
        st.write('Prediction: No Diabetes')
    else:
        st.write('Prediction: Diabetes')

if __name__ == '__main__':
    main()
