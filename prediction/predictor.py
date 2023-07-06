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
# root_chi2_selector = pickle.load(open(selector,'rb'))
comp_model = pickle.load(open('Hierarchal model/Computer/model/CompSubCat_MNB.sav','rb'))
comp_vectorizer = pickle.load(open('Hierarchal model/Computer/model/vectorizer.pkl','rb'))
comp_chi2_selector = pickle.load(open('Hierarchal model/Computer/model/selector.pkl','rb'))

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

    return text

def vectorize_text(text,tf_id_vectorizer,chi2_selector):
    vector = tf_id_vectorizer.transform([text])
    if chi2_selector != None:
        vector = chi2_selector.transform(vector)
    vector = vector.toarray()
    return vector


def website_prediction(website,dark_web):
    try:
        if dark_web:
            web = Scraper(website,dark_web=dark_web)
            text = cleaning_text(web)
        else:
            scrapTool = ScrapTool()
            web = dict(scrapTool.visit_url(website))
            text = cleaning_text(web['website_text'])
        
        vector = vectorize_text(text,root_tf_id_vectorizer,None)
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
                print("The website is under the category of Comp & Technology(others)")
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

websites = [
    ('http://vtgk7uzp7v2ba2wiv2m5diuklzszgcparux5cjxvjlt5mqaoehwjqxad.onion/',True),
    ('http://zohlm7ahjwegcedoz7lrdrti7bvpofymcayotp744qhx6gjmxbuo2yid.onion/',True),
    ('http://lockbit7z2mmiz3ryxafn5kapbvbbiywsxwovasfkgf5dqqp5kxlajad.onion',True),
    ('http://rnsm777cdsjrsdlbs4v5qoeppu3px6sb2igmh53jzrx7ipcrbjz5b2ad.onion',True),
    ('http://hacktowns3sba2xavxecm23aoocvzciaxirh3vekg2ovzdjgjxedfvqd.onion/home.php',True),
    ('http://7ukmkdtyxdkdivtjad57klqnd3kdsmq6tp45rrsxqnu76zzv3jvitlqd.onion',True),
    ('http://a6jcl5br7x77owlyx7fn2volfocesqbyddqajcatnfqebzzwbmejtbid.onion',True),
    ('http://p66slxmtum2ox4jpayco6ai3qfehd5urgrs4oximjzklxcol264driqd.onion',True),
    ('http://lockbitapt5x4zkjbcqmz6frdhecqqgadevyiwqxukksspnlidyvd7qd.onion',True),
    ('http://blog6zw62uijolee7e6aqqnqaszs3ckr5iphzdzsazgrpvtqtjwqryid.onion',True),
    ('http://breachdbsztfykg2fdaq2gnqnxfsbj5d35byz3yzj73hazydk4vq72qd.onion',True),
    ('http://uwcryspionvholmkfxoqt2xns5mvnct34ytacugxtqpqrnka2oqm6kqd.onion',True),
    ('http://5v3hgztpzes4mu3ii4mmmirufahwhsux6rgmwcmeclkedonirvpa5yad.onion',True),
    ('http://awoqb72p5kqcxtrsgoftnegnj6r33ikvq4n2fnhvdzsmuf4vzg26lcad.onion',True),
    ('http://zcashfgzdzxwiy7yq74uejvo2ykppu4pzgioplcvdnpmc6gcu5k6vwyd.onion',True),
    ('http://4p6i33oqj6wgvzgzczyqlueav3tz456rdu632xzyxbnhq4gpsriirtqd.onion',True),
    ('http://4wmicvgfju43ejudk2km3a7jhkwyewtgwwxbttcquca4fm3tilmtocyd.onion',True),
    ('http://5xtktk7q63l533qiiupxc5nvnivzzn4mvonwaajl6vdp5p7bjatkcbad.onion',True),
    ('http://7bw24ll47y7aohhkrfdq2wydg3zvuecvjo63muycjzlbaqlihuogqvyd.onion',True),
    ('http://3el34jbxuwiigfzf4o75evttgilswlp7qdwcap6rfxi3eh27dzenaaqd.onion',True),
    ('https://cardvilla.cc/',False),
    ('http://wannab666qqm2nhtkqmwwps3x2wu2bv33ayvmf4jyb6g3ibmitdzkcyd.onion/',True),

]

count = 0
for website in websites:
    print(website[0])
    website_prediction(website[0],website[1])