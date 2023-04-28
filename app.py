import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import streamlit as st
ps = PorterStemmer()



st.set_page_config(page_title='Spam Email Detector', page_icon=':email:')

hide_menu="""
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
</style>
"""
  
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
css = """
<style>
    button.css-1x8cf1d.edgvbvh10
{ background-image: linear-gradient(to right, #1A2980 0%, #26D0CE  51%, #1A2980  100%);
        margin: 10px;
        padding: 10px 35px;
        text-align: center;
        text-transform: uppercase;
        transition: 0.5s;
        background-size: 200% auto;
        color: white;            
        box-shadow: 0 0 20px #3936bb;
        border-radius: 10px;
        display: block;
}

      .css-1x8cf1d.edgvbvh10:hover 
      {
        background-position: right center;
        color: #fff;
        text-decoration: none;
 }
 
div.css-1629p8f.e16nr0p31>h2
{font-family: "Source Sans Pro", sans-serif;
        font-size: 34px !important;
        line-height: 1;
        color: #ffffff;
        border: 3px solid rgba(10, 59, 151, 0.505);
        border-radius: 10px;
        margin: 10px;
        display: block;
        text-align: center;
        letter-spacing: 0.5px;
        word-spacing: 1px;
        text-shadow: #1A2980 1px 0.5px ;
}
</style>
"""

st.markdown(page_bg_style, unsafe_allow_html=True)
st.markdown(hide_menu, unsafe_allow_html=True)
st.markdown(css, unsafe_allow_html=True)
    
    
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

st.markdown("""
<style>
.big-font {
    text-align: center;
    font-size:50px !important;
    color: white;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-font"><b>Email Classifier</b></div>', unsafe_allow_html=True)
st.subheader("")
label=" "
input_mail = st.text_area(label,placeholder="Enter your mail here")


if st.button('Classify'):

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