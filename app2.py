import streamlit as st
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize model loading in session state
@st.cache_resource
def load_model():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    tokenizer = AutoTokenizer.from_pretrained("JustinDu/BARTxiv")
    model = AutoModelForSeq2SeqLM.from_pretrained("JustinDu/BARTxiv")
    return tokenizer, model

def remove_square_brackets(text):
    # Remove anything in square brackets, including the brackets themselves
    return re.sub(r'\[[^\]]*\]', '', text)

def summarize_text(text, summary_length, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=summary_length + 50,
        min_length=summary_length,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Remove square brackets from the summary
    cleaned_summary = remove_square_brackets(summary)
    return cleaned_summary

# Load model once
tokenizer, model = load_model()

# App interface
st.title("Research Paper Summarizer")
st.markdown("""
    This tool summarizes research papers efficiently. 
    Please enter the text of the paper below and select the desired summary length.
""")

# Input text box with help tooltip
input_text = st.text_area(
    "Enter the text to summarize:",
    height=300,
    help="Paste your research paper text here. The model works best with academic text and articles."
)

# Slider with help tooltip, with step of 25
summary_length = st.slider(
    "Select summary length:",
    min_value=200,
    max_value=350,
    value=250,
    step=25,
    help="Adjust the length of your summary. Longer summaries will include more details."
)

# Generate summary
if st.button("Summarize"):
    if input_text:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text, summary_length, tokenizer, model)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
