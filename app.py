import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import streamlit as st
ps = PorterStemmer()



st.set_page_config(page_title='Spam Email Detector', page_icon=':email:')
  
page_bg_style="""
<style>
[data-testid="stAppViewContainer"]{
background-color: #0093E9;
background-image: linear-gradient(180deg, #0093E9 2%, #80D0C7 58%);
}
[data-testid="stHeader"]{
    background-color: rgba(0, 0, 0, 0);
}

</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

with open("design.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)
    
    
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

    

st.title("Spam Email Classifier")
st.subheader("")
label=" "
input_mail = st.text_area(label,placeholder="Enter your mail here")



if st.button('Predict'):

    # 1. preprocess
    transformed_mail = transform_text(input_mail)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_mail])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("This is a spam mail")
    else:
        st.header("This is not a spam mail")