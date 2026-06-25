from Bias_detector import Scraper,BiasDetector,ArticleProcessor
from flask import Flask, render_template, request

# the below line creates the flask application
app = Flask(__name__)

# once the server starts, objects are created and the model is trained immediately.
textcleaner = ArticleProcessor()
model = BiasDetector()
model.load()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error_msg = None

    if request.method == 'POST':

        URL = request.form.get('article_url')

        if URL:
            try:
                article = Scraper(URL)
                raw_text = article.scrape()
                cleaned_text = textcleaner.clean_text(raw_text)

                array_of_vectors = model.vectorizer.transform([(cleaned_text)]).toarray()[0]
                feature_names = model.vectorizer.get_feature_names_out()

                # below is the diagnostic code to check which words in the article flagged the detector using vectorizer.
                for words, count in zip(feature_names, array_of_vectors):
                    if count>0:
                        print(f"Flagged token: {words} (Count: {count})")
                    print("-----------------------")
                #the diagnostic code ends here.

                predictions = model.detect(cleaned_text)
                result = predictions[0]

            except Exception as e:
                print("Backend error"+e)
                error_msg = "There was an error processing the URL, please enter a valid URL."

    return render_template('index.html', prediction=result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)







