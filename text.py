import re , streamlit as st
from collections import defaultdict

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import heapq



def abstractive_summarizer(text, model, tokenizer, num):
    input_text = "summarize: " + text.strip().replace("\n", " ")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    summary_ids = model.generate(inputs, max_length=min(500 , num*20), min_length= min(50 , num*15), length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def nltk_tokenizer(txt, num):
    nltk.download('punkt_tab')
    nltk.download('stopwords')    
    freq = defaultdict(int)
    words = word_tokenize(txt)
    sentences = sent_tokenize(txt)
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    for word in words:
        if word not in [',', '.'] and word not in stopWords:
            w = stemmer.stem(word)
            freq[w] += 1
    
    mx = max(freq.values())
    
    for f in freq:
        freq[f] = freq[f]/mx # type: ignore
    
    sentence_scores = defaultdict(int)
    
    for i, sent in enumerate(sentences):
        
        for word in word_tokenize(sent):
            w = stemmer.stem(word)
            if w in freq.keys():
                
                sentence_scores[sent] += (freq[w]) # type: ignore
    
    final_sent = heapq.nlargest(min(num, len(sentences)), sentence_scores , key = sentence_scores.get) # type: ignore
    
    sente = [sent for sent in sentences if sent in final_sent]
    
    summary = ' '.join(sente)
    return summary
            
    
def spacy_tokenizer(txt, num):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(txt)    
    freq = defaultdict(int)
    sentences = list(doc.sents)
    stopWords = list(STOP_WORDS)
    
    for word in doc:
        word = word.lemma_
        if word not in [',', '.'] and word not in stopWords:
            freq[word] += 1
    
    mx = max(freq.values())
    
    for f in freq:
        freq[f] = freq[f]/mx # type: ignore
    
    sentence_scores = defaultdict(int)
    
    for i, sent in enumerate(sentences):
        
        for word in sent:
            w = word.lemma_
            if w in freq.keys():
                
                sentence_scores[sent] += (freq[w]) # type: ignore
    val = min(num, len(sentences))
    
    final_sent = heapq.nlargest(val, sentence_scores , key = sentence_scores.get) # type: ignore
    
    sente = [str(sent) for sent in sentences if sent in final_sent]
    
    summary = ' '.join(sente)
    return summary
                        

def apply_formatting(text, fmt):
    if fmt == "Bold":
        return f"**{text}**"
    elif fmt == "Italic":
        return f"*{text}*"
    elif fmt == "Upper":
        return text.upper()
    elif fmt == "Lower":
        return text.lower()
    return text

def main():
    st.set_page_config(
        page_title="Text Summarizer",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="auto"
    )

    st.title("Text Summarizer App")
    activities = ["Select an option", "Extract Summarization", "Abstract Summarization"]
    choice = st.sidebar.selectbox("Select Summarization Method:", activities)

    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "format_choice" not in st.session_state:
        st.session_state.format_choice = "Select an option"
    
    if "last_choice" not in st.session_state:
        st.session_state.last_choice = ""

    if choice != st.session_state.last_choice:
        st.session_state.summary_text = ""
        st.session_state.last_choice = choice

    if choice == "Select an option":
        st.warning("Select the summarization method!")

    if choice == "Extract Summarization":
        txt = st.text_area("Enter the input text", height=300)
        types = ["Select an option", "NLTK", "SPACY"]
        s_choice = st.selectbox("Summary choice", types)
        num = st.number_input("Number of sentences in summary", min_value=1, max_value=10, value=3)

        if st.button("Summarize the text"):
            if not txt.strip():
                st.warning("Enter the input text!")
                return
            if s_choice == "Select an option":
                st.warning("Select the summarizer type.")
                return
            text = re.sub(r'\[[0-9]*\]', '', txt)
            text = re.sub(r'[^\w,.\s]', '', text)
            text = re.sub(r'[A-Z]\Z', "", text)
            text = re.sub(r'\s+', ' ', text).strip()
            if s_choice == "NLTK":
                st.session_state.summary_text = nltk_tokenizer(text, int(num))
            elif s_choice == "SPACY":
                st.session_state.summary_text = spacy_tokenizer(text, int(num))

        if st.session_state.summary_text:
            st.subheader("Summary:")
            st.session_state.summary_text = st.text_area("Edit Summary", value=st.session_state.summary_text, height=200, key="summary_editor")
            ch = st.radio("Select the format option", ["Select an option", "Bold", "Italic", "Upper", "Lower"], key="format_choice")
            st.markdown("**Formatted Output:**")
            st.markdown(apply_formatting(st.session_state.summary_text, ch))
            
    # st.session_state.summary_text = ""
    if choice == "Abstract Summarization":
        txt = st.text_area("Enter the input text", height=300)
        num = st.number_input("Number of sentences in summary", min_value=1, max_value=10, value=3)
        # st.button("Summarize_the")
        # st.session_state.summary_text = ""
        flag  = 0
        if st.button("Summarize the text"):
            if not txt.strip():
                st.warning("Enter the input text!")
                return
        
            text = re.sub(r'\[[0-9]*\] | [0-9]+\.', '', txt)
            text = re.sub(r'[^\w,.\s]', '', text)
            text = re.sub(r'[A-Z]\Z', "", text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokenizer = AutoTokenizer.from_pretrained("t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            flag = 1
            st.session_state.summary_text = abstractive_summarizer(text, model, tokenizer, num)
        # print(session_state.summary_text)
        if st.session_state.summary_text:
            st.subheader("Abstractive Summary:")
            st.session_state.summary_text = st.text_area("Edit Summary", value=st.session_state.summary_text, height=200, key="abstract_editor")
            format_choice = st.radio("Select the format option", ["Select an option", "Bold", "Italic", "Upper", "Lower"], key="format_choice")
            st.markdown("**Formatted Output:**")
            st.markdown(apply_formatting(st.session_state.summary_text, format_choice))   

if __name__ == '__main__' :
    main()
