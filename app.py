import streamlit as st
import pandas as pd
import numpy as np
import itertools
import pickle
import sklearn
import time
import altair as alt
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#Nama: Diaz Islami
#NIM: A11.2020.13116
#Kelompok: BKDS02

# load dataset
processed_data = pd.read_csv("data/processed_data.csv")

# Atur Judul Page
st.set_page_config(
    page_title = "A11.2020.13116 - tugasMinggu3",
    page_icon = ":heart:"
)

# Memberi Judul Konten
st.title("Hungarian Heart Disease")
st.write("Oleh :red[**Diaz Islami - A11.2020.13116**]")

# Membuat Tab Konten
tab_model, tab_single, tab_multi, tab_show_data = st.tabs(["Model Accuracy", "Single-predict", "Multi-predict", "Display data"])

with tab_model:
    # Tab untuk menampilkan akurasi setiap model
    st.header("Model Scores")
    data_smote = pd.read_csv("data/data_smote.csv")

    # Normalisasi
    scaler = MinMaxScaler()
    X, y = data_smote.drop("target", axis=1), data_smote['target']
    X = scaler.fit_transform(X)

    # Load Model
    knn = pickle.load(open("model/knn_model.pkl", 'rb'))
    rf = pickle.load(open("model/rf_model.pkl", 'rb'))
    xgb = pickle.load(open("model/xgb_model.pkl", 'rb'))
    svm = pickle.load(open("model/svm_model.pkl", 'rb'))
    models = [knn, rf, xgb, svm]
    model_name = ["KNN", "Random Forest", "XGBoost", "SVM"]
    scores = []

    for i in models:
        y_pred = i.predict(X)
        accuracy = accuracy_score(y, y_pred)
        accuracy = round((accuracy * 100), 2)
        scores.append(accuracy)

    data_model = pd.DataFrame({'Model': model_name, 'Score': scores})

    # Bar Chart
    bar_chart = alt.Chart(data_model).mark_bar(color='green').encode(
        x=alt.X('Model:N', title='Model', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Score:Q', title='Score'),
        tooltip=['Model:N', 'Score:Q'],
    )

    # Menampilkan bar chart di Streamlit
    st.altair_chart(bar_chart, use_container_width=True)


with tab_single:
    # Tab untuk melakukan prediksi input user (tunggal)
    # Membuat Sidebar Untuk Input Pengguna
    st.sidebar.header("User Input")

    # Input Age
    age = st.sidebar.number_input(label=":violet[**Age**]",
    min_value=28.0, max_value=66.0)
    st.sidebar.write(f":orange[**Min**] value: :orange[**28.0**],\
    :red[**Max**] value: :red[**66.0**]")
    st.sidebar.write("")

    # Input Sex
    sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", 
    options=["Male", "Female"])
    if sex_sb == "Female": sex_sb = 0
    elif sex_sb == "Male": sex_sb = 1
    st.sidebar.write("")
    st.sidebar.write("")

    # Input CP
    cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", 
    options=["Typical angina", 
    "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    if cp_sb == "Typical angina": cp_sb = 1
    elif cp_sb == "Atypical angina": cp_sb = 2
    elif cp_sb == "Non-anginal pain": cp_sb = 3
    elif cp_sb == "Asymptomatic": cp_sb = 4
    st.sidebar.write("")
    st.sidebar.write("")

    # Input Trestbps
    trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]",
    min_value=92.0, max_value=200.0)
    st.sidebar.write(f":orange[**Min**] value: :orange[**92.0**],\
    :red[**Max**] value: :red[**200.0**]")
    st.sidebar.write("")

    # Chol
    chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]",
    min_value=85.0, max_value=603.0)
    st.sidebar.write(f":orange[**Min**] value: :orange[**85.0**],\
    :red[**Max**] value: :red[**603.0**]")
    st.sidebar.write("")

    # Input FBS
    fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 in mg/dl?**]", 
    options=["True", "False"])
    if fbs_sb == "False": fbs_sb = 0
    elif fbs_sb == "True": fbs_sb = 1
    st.sidebar.write("")
    st.sidebar.write("")

    # Input Restecg
    restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results?**]", 
    options=["Normal", 
    "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
    if restecg_sb == "Normal": restecg_sb = 0
    elif restecg_sb == "Having ST-T wave abnormality": restecg_sb = 1
    elif restecg_sb == "Showing left ventricular hypertrophy": restecg_sb = 2
    st.sidebar.write("")
    st.sidebar.write("")

    # Input Thalach
    thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]",
    min_value=82.0, max_value=190.0)
    st.sidebar.write(f":orange[**Min**] value: :orange[**82.0**],\
    :red[**Max**] value: :red[**190.0**]")
    st.sidebar.write("")
    
    # Input Exang
    exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", 
    options=["Yes", "No"])
    if exang_sb == "No": exang_sb = 0
    elif exang_sb == "Yes": exang_sb = 1
    st.sidebar.write("")
    st.sidebar.write("")

    # Input Oldpeak
    oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]",)
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    # Input Model Prediksi
    model_sb = st.selectbox(label=":red[**Model Prediksi**]", 
    options=["KNN", "Random Forest", "XGBoost", "SVM"])
    if model_sb == "KNN": model_sb = models[0] #pickle.load(open("model/knn_model.pkl", "rb"))
    elif model_sb == "Random Forest": model_sb = models[1] #pickle.load(open("model/rf_model.pkl", "rb"))
    elif model_sb == "XGBoost": model_sb = models[2] #pickle.load(open("model/xgb_model.pkl", "rb"))
    elif model_sb == "SVM": model_sb = models[3] #pickle.load(open("model/svm_model.pkl", "rb"))

    st.header("User Input as DataFrame")

    data = {
        "age":age,
        "sex":sex_sb,
        "cp":cp_sb,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs_sb,
        "restecg":restecg_sb,
        "thalach":thalach,
        "exang":exang_sb,
        "oldpeak":oldpeak
    }
    df = pd.DataFrame(data, index=['Input:'])
    st.write("")
    st.dataframe(df.iloc[:, :])
    st.write("")

    # Membuat Button Prediksi
    predict_btn = st.button("Predict", type="primary")
    st.write("")
    result = ":orange[Belum ada hasil]"
    if predict_btn:
        input_user = [[age, sex_sb, cp_sb, trestbps, chol, fbs_sb, restecg_sb, thalach, exang_sb, oldpeak]]
        input_user = MinMaxScaler().fit_transform(input_user)
        result = model_sb.predict(input_user)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        label_pred = {
            0: ":green[**Healthy**]",
            1: ":orange[**Heart disease level 1**]",
            2: ":orange[**Heart disease level 2**]",
            3: ":red[**Heart disease level 3**]",
            4: ":red[**Heart disease level 4**]",
        }
        result = label_pred.get(int(result)) # Mendapatkan Label Prediksi

    st.write("")
    st.subheader("Prediction result:")
    st.subheader(result)

with tab_multi:
    # Tab untuk melakukan multi-prediksi
    st.header("Multi-predict:")

    sample_csv = processed_data.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:

        # Input Model Prediksi
        model_sb_multi = st.selectbox(label=":red[**Pilih Model:**]", 
        options=["KNN", "Random Forest", "XGBoost", "SVM"])
        if model_sb_multi == "KNN": model_sb_multi = models[0] #pickle.load(open("model/knn_model.pkl", "rb"))
        elif model_sb_multi == "Random Forest": model_sb_multi = models[1] #pickle.load(open("model/rf_model.pkl", "rb"))
        elif model_sb_multi == "XGBoost": model_sb_multi = models[2] #pickle.load(open("model/xgb_model.pkl", "rb"))
        elif model_sb_multi == "SVM": model_sb_multi = models[3] #pickle.load(open("model/svm_model.pkl", "rb"))

        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model_sb_multi.predict(uploaded_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            label_pred = {
                    0: "Healthy",
                    1: "Heart disease level 1",
                    2: "Heart disease level 2",
                    3: "Heart disease level 3",
                    4: "Heart disease level 4",
                }
            result = label_pred.get(prediction)
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)

with tab_show_data:
    # Tab untuk menampilkan data yang telah diproses
    st.header("Display Processed Data")
    show_btn = st.button("Display", type="primary")
    if show_btn:
        st.write(processed_data[:100])
