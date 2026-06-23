from Bias_detector import X_raw,Scraper,BiasDetector,ArticleProcessor, y
from flask import Flask, render_template, request

# the below line creates the flask application
app = Flask(__name__)

# once the server starts, objects are created and the model is trained immediately.
textcleaner = ArticleProcessor()
bias_detector = BiasDetector()
bias_detector.train(X_raw,y)

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
                predictions = bias_detector.detect(cleaned_text)

                result = predictions[0]

            except Exception as e:
                print("Backend error"+e)
                error_msg = "There was an error processing the URL, please enter a valid URL."

    return render_template('index.html', prediction=result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)







