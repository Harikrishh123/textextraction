import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import re , streamlit as st
from collections import defaultdict

import torch
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from transformers import T5ForConditionalGeneration, T5Tokenizer
import heapq

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")



def abstractive_summarizer(text, num):

    input_text = "summarize: " + text.strip().replace("\n", " ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, max_length=max(500 , num*20), min_length= min(100 , num*15), length_penalty=2.0, num_beams=4, early_stopping=True)

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
    # print(choice)
    
    if choice == "Select an option":
        st.warning("Select the summarization method!")
        
    if choice == "Extract Summarization":

        txt = st.text_area("Enter the input text", height=300)
        text = re.sub(r'\[[0-9]*\]', '' , txt)
        text = re.sub(r'[^\w,.\s]', '', text)
        text = re.sub(r'[A-Z]\Z', "", text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        types = ["Select an option", "NLTK", "SPACY"]
        s_choice = st.selectbox("Summary choice", types)
        num = st.number_input("Number of sentences in summary", min_value=1 , max_value=10 , value=3)
        # print(num)
        if st.button("Summarize the text"):
            if not txt.strip():
                st.warning("Enter the input text!")
                return
            if s_choice == "Select an option":
                st.warning("Select the option above!")
            
            if s_choice == "NLTK":
                summary = nltk_tokenizer(text, int(num))
                st.header("Summary using NLTK:")
                # print(summary)
                st.write(summary)
            
            if s_choice == "SPACY":
                summary = spacy_tokenizer(text, int(num))
                # print("Hello")
                st.header("Summary using SPACY:")
                st.write(summary)
    
    if choice == "Abstract Summarization":
        txt = st.text_area("Enter the input text", height=300)
        
        text = re.sub(r'\[[0-9]*\]', '' , txt)
        text = re.sub(r'[^\w,.\s]', '', text)
        text = re.sub(r'[A-Z]\Z', "", text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        num = st.number_input("Number of sentences in summary", min_value=1 , max_value=10 , value=3)
        
        if st.button("Summarize the text"):
            
            if not txt.strip():
                st.warning("Enter the input text!")
                return
            summary = abstractive_summarizer(text, num)
            st.header("Abstractive Summary :")
                # print(summary)
            st.write(summary)
              

if __name__ == '__main__' :
    main()

# txt = '''
# Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
# '''

# print(nltk_tokenizer(txt, 3))
    
