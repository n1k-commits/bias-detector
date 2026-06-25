import pandas as pd
import requests
from newspaper import Article
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib



class ArticleProcessor:
    def __init__(self):

        pass

    def clean_text(self, unmodified_text):
        words = str(unmodified_text).lower().split()
        return ' '.join(words)

class Scraper:
    def __init__(self,url):

        self.url = url
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,te;q=0.7',
            'Referer': 'https://www.google.com/'}

    def scrape(self):

        response = requests.get(self.url,headers=self.headers,timeout=10)

        if response.status_code != 200:
            raise Exception(f"Blocked by firewall! Status: {response.status_code}")

        source = Article(response.url)
        source.set_html(response.text)
        source.parse()
        return source.text

class BiasDetector:
    def __init__(self):

        self.model = MultinomialNB()
        self.is_trained = False

    def train(self, raw_data, labels, fixed_vocabulary):
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', vocabulary=fixed_vocabulary)
        X_final = self.vectorizer.fit_transform(raw_data)
        self.model.fit(X_final, labels)
        self.is_trained = True
        self.save()

    def save(self):
        if self.is_trained:
            joblib.dump(self.vectorizer,'bias_vectorizer.joblib')
            joblib.dump(self.model,'bias_model.joblib')
            print("The model has been trained and successfully saved")
        else:
            print("Cannot save a model that has not been trained")

    def load(self):
        try:
            self.vectorizer = joblib.load('bias_vectorizer.joblib')
            self.model = joblib.load('bias_model.joblib')
            self.is_trained = True
            print("The model has been loaded and is up and running.")
        except FileNotFoundError:
            print("Critical files of the model are missing! Train the model before proceeding.")


    def detect(self,clean_article):
        vectorized_article = self.vectorizer.transform([clean_article])
        return self.model.predict(vectorized_article)

#a dictionary with words typically used in articles of respective political leanings.

political_leanings = {'Left':['dalit','equality','uppercaste','welfare','muslim','dalits','muslims','oppress','subsidy','subsidies','socialist','saffron','RSS','BJP','Brahminical','Brahmin','secular','minority','minorities','oppressed','progressive','marginalised','privileged','privilege','undocumented migrants','human rights'],
                      'Center':['implemented','stated','announced','reports','officials','official','statements','data','statistical','statistically','parliamentary','regulatory','constitutional','bureaucratic','scheduled'],
                      'Right':['nationalist','terrorist','religious','privatize','growth','immigrants','naxals','separatists','minority','appeasement','free market','capitalist','infrastructure','projects','nations interest','national interest','security','indigenous','narrative','woke','riots','biased','disruptive','anti-national']}


final_training_set = []
for ideology,words in political_leanings.items():
    for word in words:
        final_training_set.append({'Ideology':ideology,'Text':(word+" ")*5})

# print(final_training_set)

df = pd.DataFrame(final_training_set)

X_raw = df['Text']
y = df['Ideology']

