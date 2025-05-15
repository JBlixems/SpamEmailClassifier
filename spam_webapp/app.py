import csv
import sys
from flask import Flask, render_template, request
from markupsafe import Markup
import joblib
import re

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

vectorizer, model = joblib.load("spam_model.joblib").values()

app = Flask(__name__)

def restore_commas(text):
    return text.replace('|', ',')


def explain_spam(text, vectorizer, model, top_n=8):
    X_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    contributions = [(feature_names[idx], tfidf * coefs[idx])
                     for idx, tfidf in zip(X_vec.indices, X_vec.data)]
    top_tokens = {tok for tok, score in sorted(contributions, key=lambda x: x[1], reverse=True)[:top_n]}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, top_tokens)) + r")\b", flags=re.IGNORECASE)
    def repl(m):
        w = m.group(0)
        return f"<span class='spam-highlight'>{w}</span>" if w.lower() in top_tokens else w
    return pattern.sub(repl, text)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    highlighted_text = None

    if request.method == 'POST':
        raw = request.form['email_text']

        text = restore_commas(raw)
        X_vec = vectorizer.transform([text])

        pred_num = model.predict(X_vec)[0]
        prediction = {1: 'spam', 0: 'human'}.get(pred_num, 'unknown')

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_vec)[0]
            idx = list(model.classes_).index(pred_num)
            confidence = proba[idx] * 100

        highlighted_text = explain_spam(text, vectorizer, model)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           highlighted_text=Markup(highlighted_text) if highlighted_text else None)

if __name__ == '__main__':
    app.run(debug=True)