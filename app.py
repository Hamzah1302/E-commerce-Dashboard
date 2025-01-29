import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# st.markdown(
#     """
# # My first app
# Hello, para calon praktisi data masa depan
# """
# )

# st.title('Belajar Data Science')
# st.header('Belajar Analisi data')
# st.subheader('Belajar Machine Learning')

# st.caption('Copyright (c) Cipta Muda AI')

# code = """def hello():
#     print("Hello, Streamlit")"""
# st.code(code,language='python')
# st.text("Aku akan jadi orang kaya")
# st.latex(r"""
#     \sum_{k=0}^{n-1} ar^k =
#     a \left(\frac{1-r^{n}}{1-r}\right)
# """)

# df = pd.DataFrame({
#     'c1' : [1,2,3,4],
#     'c2' : [10,20,30,40],
# })

# st.dataframe(data=df, width=500, height=150)


# df = pd.DataFrame({
#     'c1' : [1,2,3,4],
#     'c2' : [10,20,30,40],
# })
# st.table(data=df)


# st.metric(label='Temperature', value="28 °C", delta="1.2 °C")

# st.json ({
#     'c1' : [1,2,3,4],
#     'c2' : [10,20,30,40],
# })

# x = np.random.normal(15, 5, 250)

# fig, ax = plt.subplots()
# ax.hist(x=x, bins=15)

# st.pyplot(fig)


 
# uploaded_file = st.file_uploader('Choose a CSV file')
 
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.dataframe(df)

# name = st.text_input(label='Nama lengkap', value='')
# st.write('Nama: ', name)


# text = st.text_area('Feedback')
# st.write('Feedback: ', text)
# age = st.number_input("Masukan Usia Anda:", min_value=0, max_value=100)
# st.write(f"Usia Anda : {age}")

# number =st.number_input(label='Umur')
# st.write('Umur: ', int (number), 'tahun')

# date = st.date_input(label='Tanggal Lahir', min_value=datetime.date(1930, 1, 1))
# st.write('Tanggal lahir:', date)

# picture = st.camera_input('Take a picture')
# if picture:
#     st.image(picture)

# values = st.slider(
#     label='Select a range of values',
#     min_value=0, max_value=100, value=(0, 100))
# st.write('Values:', values)

# st.title('Belajar Analisis data ')
# with st.sidebar:
#     st.text('Ini merupakan Sidebar')
#     values = st.slider(
#         label='select a range of values',
#         min_value=0, max_value=100, value=(0, 100)

#     )
#     st.write('values:', values)

# st.title('Belajar Analisis data ')
# col1, col2, col3 = st.columns([2,1,1])

# with col1:
#     st.header("kolom1")
#     st.image("https://static.streamlit.io/examples/cat.jpg")
# with col2:
#     st.header("kolom2")
#     st.image("https://static.streamlit.io/examples/dog.jpg")
# with col3:
#     st.header("kolom3")
#     st.image("https://static.streamlit.io/examples/owl.jpg")

# st.title('Belajar Analisis data ')
# tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

# with tab1:
#     st.header("kolom1")
#     st.image("https://static.streamlit.io/examples/cat.jpg")
# with tab2:
#     st.header("kolom2")
#     st.image("https://static.streamlit.io/examples/dog.jpg")
# with tab3:
#     st.header("kolom3")
#     st.image("https://static.streamlit.io/examples/owl.jpg")


# with st.container():
#     st.write("Inside the container")
#     x = np.random.normal(15, 5, 250)

#     fig, ax = plt.subplots()
#     ax.hist(x=x, bins=15)
#     st.pyplot(fig)
# st.write("outside the container")




# x = np.random.normal(15, 5, 250)
 
# fig, ax = plt.subplots()
# ax.hist(x=x, bins=15)
# st.pyplot(fig) 
# with st.expander("See explanation"):
#     st.write(
#         """Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
#         sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
#         Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris 
#         nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor 
#         in reprehenderit in voluptate velit esse cillum dolore eu fugiat 
#         nulla pariatur. Excepteur sint occaecat cupidatat non proident, 
#         sunt in culpa qui officia deserunt mollit anim id est laborum.
#         """
#     )

# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# Cek apakah file ada
import os
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Cek file di direktori
st.write("Files in directory:", os.listdir())

# Load model dan vectorizer
model = joblib.load('random_forest_model.pkl')  # Pastikan file ini ada
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Pastikan file ini ada
# Contoh HTML dan CSS
st.markdown("""
    <style>
    .custom-title {
        color: blue;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    <h1 class="custom-title">Selamat Datang di Aplikasi Streamlit!</h1>
    <p style="color: gray; font-size: 20px;">Ini adalah teks dengan gaya kustom menggunakan HTML dan CSS.</p>
""", unsafe_allow_html=True)
# Streamlit App
st.title('Sentiment Analysis')
text_input = st.text_area('Masukkan teks review')  # Input teks dari user

if st.button('Prediksi Sentimen'):
    if text_input.strip():  # Pastikan input tidak kosong
        # Vectorisasi input
        text_vectorized = vectorizer.transform([text_input])
        
        # Prediksi sentimen
        prediction = model.predict(text_vectorized)[0]
        
        # Tampilkan hasil prediksi
        st.write(f'Sentimen: {prediction}')
    else:
        st.write("Masukkan teks untuk prediksi!")
