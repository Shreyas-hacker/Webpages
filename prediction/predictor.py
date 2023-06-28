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
model = pickle.load(open('model_MNB/model/RootModel.sav','rb'))
tf_id_vectorizer = pickle.load(open(vectorizer,'rb'))
chi2_selector = pickle.load(open(selector,'rb'))

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
def cleaning_text(text):
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
    vector = tf_id_vectorizer.transform([text])
    vector = chi2_selector.transform(vector)
    vector = vector.toarray()
    return vector


def website_prediction(website,model):
    scrapTool = ScrapTool()
    try:
        # web = dict(scrapTool.visit_url(website))
        # text = cleaning_text(web['website_text'])
        web = Scraper(website)
        text = cleaning_text(web)
        prediction = model.predict(text)
        web_cat = prediction[0]
        category = ""
        if web_cat == 0:
            category = 'Adult'
        elif web_cat == 1:
            category = 'Business/Corporate'
        elif web_cat == 2:
            category = 'Computers and Technology'
        elif web_cat == 3:
            category = 'E-commerce'
        elif web_cat == 4:
            category = 'Financial Crime'
        elif web_cat == 5:
            category = "Forums"
        elif web_cat == 6:
            category = "Law and Government"
        elif web_cat == 7:
            category = "Narcotics"
        elif web_cat == 8:
            category = "News"
        elif web_cat == 9:
            category = "Social Media"
        print(f'The website is under the category of {category}')

    except Exception as e:
        print(e)
        print("Connection Timeout")


# website_prediction('http://yxkdzgrty3hqlhpr37sqma5yujlsmcxtrfjgqxyms5cwnmirz62ck7qd.onion',model)