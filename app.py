from flask import Flask, jsonify, request
from googletrans import Translator
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

translator = Translator()
summarizer = pipeline("summarization")
analyzer = SentimentIntensityAnalyzer()


@app.route('/')
def welcome():
    return 'Welcome to my sentiment analysis machine!'


@app.route('/sentiment-analysis')
def sentiment_analysis():
    data = request.get_json()
    text_id = data.get('content', '')

    text_en = translator.translate(text_id, dest='en').text

    summary_result = summarizer(text_en, max_length=len(text_id), min_length=25, do_sample=False)
    summary_en = summary_result[0]['summary_text']
    summary_id = translator.translate(summary_en, dest='id').text

    scores = analyzer.polarity_scores(text_en)
    sentiment_score = scores['compound']
    sentiment = "positif" if sentiment_score >= 0 else "negatif"

    return jsonify({
        "code": 200,
        "message": "Get sentiment analysis successfully",
        "data": {
            "sentiment": sentiment,
            "summary": summary_id
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=7070)
