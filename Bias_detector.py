import pandas as pd
import requests
from newspaper import Article
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class ArticleProcessor:
    def __init__(self):

        # below are the words that are usually considered useless for determining bias because they are considered stopwords in natural language processing

        self.stopwords = {'a','as','am','are','and','at','also','all','after','against','any',"aren't",'any','be','because','before','being','but','buy','before','can','could',"can't",'cannot','did','do','does',"didn't",'doing',
                          'during','down','each','for','far','few','further','farther','has','had','have','he','him','her','how','is','it','if','in','i',',I','into','its','me','more','most','much','my','no','nor','not','of','or',
                          'on','off','only','our','other','over','out','ok','so','some','such','to','this','that','these','those','they','them','then','than','too','through','us','under','until','up','when','where','was','were',
                          'what','we','why','who','whom','would','you','yours','your'}

    def clean_text(self, unmodified_text):
        words = str(unmodified_text).lower().split()

        #the below for loop selectively takes all the words from the article that are not stop words and returns it as a full string.

        clean_words = []
        for word in words:
            if word not in self.stopwords:
                clean_words.append(word)

        return ' '.join(clean_words)

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

class BiasDetector:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        self.model = MultinomialNB()
        self.is_trained = False

    def train(self, raw_data, labels):
        X_final = self.vectorizer.fit_transform(raw_data)
        self.model.fit(X_final, labels)
        self.is_trained = True

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


# BiasModel = BiasDetector()
# # article = 'https://timesofindia.indiatimes.com/city/delhi/18-charged-in-uttam-nagar-holi-murder-2-minors-also-involved/articleshow/130841569.cms'
#
# Websource = Scraper(article)
#
# raw_article = Websource.scrape()
#
# cleaner = ArticleProcessor()
#
# BiasModel.train(X_raw,y)
#
# processed_article = cleaner.clean_text(raw_article)
#
# predictions = BiasModel.detect(processed_article)
#
# print("Political leaning: "+predictions[0])