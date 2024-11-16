from upscale_image import upscale_image, load_image, exceed_resolution_threshold
from load_model import load_model
import streamlit as st

st.set_page_config(layout="centered", 
                   page_title="Image Upscaler")

MAX_SIZE =  1024 * 1024 #1 MB
extensions = ["png","jpg","jpeg"]
rcan = load_model('src/rcan2.pth')


st.write('## RCAN Super Resolution')
st.write("""This project implements a deep learning model based on RCAN (Residual Channel Attention Network) 
for image super-resolution (4x upscale).""")
st.write('*Note: this model is slow and still experimental.*')
st.write('*Note: because of limited computation resource, only support image up to 122500 pixels (height x width).*')

st.image('static/concatenated.png',caption='Original vs Upscaled')

upload=st.file_uploader(
    "Upload an image",
    type=extensions,
    accept_multiple_files=False,
)


if upload:
    if upload.size > MAX_SIZE:
        st.error('File is too large, please upload an smaller image')
    else:
        image = load_image(upload)
        if exceed_resolution_threshold(image):
            st.error('Image resolution is too large, please upload an smaller image')
        else:
            out_bytes = upscale_image(image,rcan)
            filename:str = upload.name
            for ext in extensions:
                if filename.endswith(ext):
                    filename = filename.replace(ext,'')
            st.download_button(
                'Download the Upscaled Image',
                data=out_bytes,
                file_name=f'{filename}_scaled.png',
                mime='image/png'
            )



