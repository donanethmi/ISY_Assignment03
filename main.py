import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("pickle/model.h5")

tokenizer_file = "pickle/token.pkl"
with open(tokenizer_file, "rb") as f:
    tokenizer = pickle.load(f)

def text_to_vector(text,token):
  sequences_test = token.texts_to_sequences([text])
  sequs_matrics_test = pad_sequences(sequences_test,maxlen = 1000)
  return sequs_matrics_test 

st.title("Sentiment Analysis")
text = st.text_input("Type here your Essay topic")

submitted = st.button("submit")

if submitted and text:
    with st.spinner("Processing..."):
        processed_text = text_to_vector(text,tokenizer)

        prediction = model.predict(processed_text)[0][0]
        
        sentiment = "positive" if prediction > 0.5 else "negative"

    st.write("Sentiment:", sentiment)


