import pickle
import numpy as np
import re
import sklearn as sk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from darkwebScrapper import Scraper

stop_words = stopwords.words('english')
lem = WordNetLemmatizer()

loaded_model = pickle.load(open('./model/CompSubCat_SVM.sav', 'rb'))
root_model = pickle.load(open('../../model_MNB/model/RootModel.sav', 'rb'))
root_model_tfidf = pickle.load(open('../../model_MNB/model/vectorizer.pkl', 'rb'))
root_model_chi2_selector = pickle.load(open('../../model_MNB/model/selector.pkl', 'rb'))
tfidf = pickle.load(open('./model/vectorizer.pkl', 'rb'))
chi2_selector = pickle.load(open('./model/selector.pkl', 'rb'))

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a'
    elif nltk_tag.startswith('V'):
        return 'v'
    elif nltk_tag.startswith('R'):
        return 'r'
    else:
        return 'n'
    
#Define function to lemmatize each word with its POS tag
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text)
    pos_tagged_text = [(word, pos_tagger(pos_tag)) for word, pos_tag in pos_tagged_text]
    return [lem.lemmatize(word, pos_tag) for word, pos_tag in pos_tagged_text]

def cleaning(text,tfidf,chi2_selector):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)
    text = re.sub(r'\s+',' ',text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word)>3]
    text = lemmatize_words(text)
    text = ' '.join(text)
    vector = tfidf.transform([text])
    vector = chi2_selector.transform(vector)
    vector = vector.toarray()
    
    return vector


def prediction(website,dark_web):
    try:
        website_text = Scraper(website,dark_web)
        vector = cleaning(website_text,root_model_tfidf,root_model_chi2_selector)
        prediction = root_model.predict(vector)
        pred_cat = prediction[0]
        category = ""
        if pred_cat == 0:
            categor
        vector_subcat = cleaning(website_text,tfidf,chi2_selector)
        prediction_subcat = loaded_model.predict(vector_subcat)
        pred_subcat = prediction_subcat[0]
        subcategory = ""
        if pred_subcat == 0:
            subcategory = 'Computers and Technology'
        elif pred_subcat == 1:
            subcategory = 'Cryptocurrency'
        else:
            subcategory = 'Cyber Security'
        print(f'The website is under the category of {subcategory}')
    except Exception as e:
        print("Something went wrong!")