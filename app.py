import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import SVC for SVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

 
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


st.title("Multiple Disease Prediction APP")


lottie_url = "https://lottie.host/53e1a6b6-2124-4ac5-840c-36c3e8e55193/1vhH08mCWG.json"
lottie_json = load_lottie_url(lottie_url)
st_lottie(lottie_json, speed=1, width=700, height=200, key="initial")


# Menu to choose between "Diabetes" and "Heart Disease"
selected_disease = st.selectbox("Choose Disease to Predict", ("Diabetes", "Heart Disease"))


# File uploader
uploaded_file = st.file_uploader(f"Choose a CSV file for {selected_disease} Prediction", type=["csv"])


if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(data.head())  # showing few data in the frame
    

    if selected_disease == "Diabetes":
        necessary_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    else:
        necessary_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


    if all(col in data.columns for col in necessary_columns):
        # Separating features and target
        X = data[necessary_columns[:-1]]  
        y = data[necessary_columns[-1]]  
        

        # Spliting training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        # normalizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        

        # Train an SVM model
        model = SVC(kernel='linear')  
        model.fit(X_train, y_train)
        

        # Evaluating the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Trained with Accuracy: {accuracy * 100:.2f}%")
        

        #new predictions
        st.subheader(f'{selected_disease} Prediction')
        
        if selected_disease == "Diabetes":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Pregnancies = st.number_input('Number of Pregnancies', min_value=0, format='%d')
            
            with col2:
                Glucose = st.number_input('Glucose Level', min_value=0, format='%d')
            
            with col3:
                BloodPressure = st.number_input('Blood Pressure value', min_value=0, format='%d')
            
            with col1:
                SkinThickness = st.number_input('Skin Thickness value', min_value=0, format='%d')
            
            with col2:
                Insulin = st.number_input('Insulin Level', min_value=0, format='%d')
            
            with col3:
                BMI = st.number_input('BMI value', min_value=0.0, format='%.2f')
            
            with col1:
                DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format='%.2f')
            
            with col2:
                Age = st.number_input('Age of the Person', min_value=0, format='%d')

            
            
            diab_diagnosis = ''
            

           
            if st.button('Diabetes Test Result', type="primary"):
                inputs = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
                inputs = scaler.transform(inputs)  # normalizing user inputs
                diab_prediction = model.predict(inputs)
                
                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                    
                st.success(diab_diagnosis)
        
        elif selected_disease == "Heart Disease":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input('Age', min_value=0, format='%d')
            
            with col2:
                sex = st.selectbox('Sex', [0, 1])  
            
            with col3:
                cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])  
            
            with col1:
                trestbps = st.number_input('Resting Blood Pressure', min_value=0, format='%d')
            
            with col2:
                chol = st.number_input('Cholesterol Level', min_value=0, format='%d')
            
            with col3:
                thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, format='%d')
            
            with col1:
                oldpeak = st.number_input('Oldpeak (ST Depression)', min_value=0.0, format='%.2f')
            
            with col2:
                exang = st.selectbox('Exercise Induced Angina', [0, 1])  
            
           
            heart_diagnosis = ''
            
            

            if st.button('Heart Disease Test Result', type="primary"):
                inputs = [[age, sex, cp, trestbps, chol, thalach, exang, oldpeak]]
                inputs = scaler.transform(inputs)  # normalize user inputs
                heart_prediction = model.predict(inputs)
                
                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person has heart disease'
                else:
                    heart_diagnosis = 'The person does not have heart disease'
                    
                st.success(heart_diagnosis)
    else:
        st.error(f"The dataset does not contain the necessary columns for {selected_disease}.")
else:
    st.write(f"Please upload a CSV file for {selected_disease} prediction.")



page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Hide the Streamlit menu (three dots) & footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(hide_menu_style, unsafe_allow_html=True)
