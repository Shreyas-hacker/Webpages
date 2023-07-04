from ScrapTool import ScrapTool
from darkwebScrapper import Scraper
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import pickle

vectorizer = 'model_MNB/model/vectorizer.pkl'
selector = 'model_MNB/model/selector.pkl'
lem = WordNetLemmatizer()
stop_words = stopwords.words('english')
root_model = pickle.load(open('model_MNB/model/RootModel.sav','rb'))
root_tf_id_vectorizer = pickle.load(open(vectorizer,'rb'))
root_chi2_selector = pickle.load(open(selector,'rb'))
comp_model = pickle.load(open('../Hierarchal model/Computer/model/CompSubCat_SVM.sav','rb'))
comp_vectorizer = pickle.load(open('../Hierarchal model/Computer/model/vectorizer.pkl','rb'))
comp_chi2_selector = pickle.load(open('../Hierarchal model/Computer/model/selector.pkl','rb'))

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

#cleaning text and preprocessing
def cleaning_text(text,tf_id_vectorizer,chi2_selector):
    text = text.lower()
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(r'_+',' ',text)
    text = re.sub(r'\d+','',text)
    text = re.sub(r'\s+',' ',text)   
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word)>3]
    text = lemmatize_words(text)
    # text = [lem.lemmatize(word) for word in text]
    text = ' '.join(text)

def vectorize_text(text,tf_id_vectorizer,chi2_selector):
    vector = tf_id_vectorizer.transform([text])
    vector = chi2_selector.transform(vector)
    vector = vector.toarray()
    return vector


def website_prediction(website,dark_web):
    try:
        if dark_web==False:
            scrapTool = ScrapTool()
            web = dict(scrapTool.visit_url(website))
            text = cleaning_text(web['website_text'])
        else:
            web = Scraper(website,dark_web)
            text = cleaning_text(web)
        vector = vectorize_text(text,root_tf_id_vectorizer,root_chi2_selector)
        prediction = root_model.predict(vector)
        web_cat = prediction[0]
        category = ""

        if web_cat == 0:
            category = 'Adult'
        elif web_cat == 1:
            category = 'Computers and Technology'
            vector = vectorize_text(text,comp_vectorizer,comp_chi2_selector)
            prediction = comp_model.predict(vector)
            web_cat = prediction[0]
            if web_cat == 0:
                print("The website is under the category of Computers and Technology")
            elif web_cat == 1:
                print("The website is under the category of Cryptocurrency")
            else:
                print("The website is under the category of Cyber Security")
            return
        elif web_cat == 2:
            category = 'Financial Crime'
        elif web_cat == 3:
            category = 'Forums'
        elif web_cat == 4:
            category = "Intelligence"
        elif web_cat == 5:
            category = "Law and Government"
        elif web_cat == 6:
            category = "Marketplace"
        elif web_cat == 7:
            category = "Narcotics"
        elif web_cat == 8:
            category = "News"
        else:
            category = "Social Media"
        print(f'The website is under the category of {category}')

    except Exception as e:
        print(e)
        print("Connection Timeout")


website_prediction('http://yxkdzgrty3hqlhpr37sqma5yujlsmcxtrfjgqxyms5cwnmirz62ck7qd.onion',model)