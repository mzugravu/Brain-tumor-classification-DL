import streamlit as st
from PIL import Image
from clf import classify



hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

img = Image.open("logo.jpg")
st.image(img, width=300)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("BRA'INSA App")
st.write("")

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    label = classify(file_up)

    # print out the prediction
    html_str = f"""
    <style>
    p.a {{
      font: bold 50px Courier;
    }}
    </style>
    <p class="a">Prediction is: {label}</p>
    """

    st.markdown(html_str, unsafe_allow_html=True)
