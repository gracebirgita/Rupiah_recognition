import os
import numpy 
import streamlit as st
from PIL import Image
from ImageTOTEXT import image_to_text
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from IndoSpeechT5 import create_speaker_embedding_from_wav, replace_numbers_with_words, cleanup_text, normalize_text, text_to_speech, SpeechT5HifiGan
import torch
import soundfile as sf
import re

#for logic
pred = {}
huruf_pred = {}
total = {}

# Make sure the folder exists
os.makedirs("image", exist_ok=True)

st.title("Money Detection App")
st.header("Welcome!")
st.write("Please upload a picture")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button('Scan Image with YOLO and Generate Voice with TTS'):
        # Define a path to save the file
        image_path = os.path.join("image", uploaded_file.name)
        
        # Save the uploaded file
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert backslashes to forward slashes (just in case, for Windows compatibility)  
        image_path = image_path.replace("\\", "/")
        
        # Run the YOLO prediction
        pred, huruf_pred, total = image_to_text(image_path)

        # st.write(f"Predicted digits count: {pred}")
        # st.write(f"Predicted letters count: {huruf_pred}")
        st.write(f"Total sum: {total}")

        if total:
            with st.spinner("Generating speech..."):
                # Generate and immediately play audio
                audio_bytes = text_to_speech(f"{total} Rupiah")
                st.audio(audio_bytes, format='audio/wav')
        else:
            st.write(f"Not detecting number on image")
            st.write(f"please re-capture your image")
            ##with open(audio_file, "rb") as f:
                ##st.download_button(
                    ##label="Download Audio",
                    ##data=f,
                    ##file_name="currency_value.wav",
                    ##mime="audio/wav"
                ##
                
           
        
        
        
        
            
            
