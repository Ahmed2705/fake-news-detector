import joblib
import spacy
import nltk
from nltk.corpus import stopwords
import gradio as gr
import numpy as np

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -------------------------------
# Text Preprocessing
# -------------------------------
def lemmatize_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(lemmas)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_news(text):
    if not text.strip():
        return "⚠️ Please enter some text.", ""

    clean_text = lemmatize_text(text)
    X_input = vectorizer.transform([clean_text])
    pred_score = model.decision_function(X_input)[0]
    prediction = model.predict(X_input)[0]

    confidence = 1 / (1 + np.exp(-abs(pred_score)))
    confidence = round(confidence * 100, 2)

    label = "🟩 Real News" if prediction == 1 else "🟥 Fake News"
    return label, f"{confidence}% Confidence"

# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    # Header
    gr.HTML(
        """
        <div style="
            text-align:center;
            font-size: 2.2em;
            font-weight: bold;
            color: white;
            background: linear-gradient(90deg, #0057ff, #00c2ff);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            ">
            📰 Fake News Detector
        </div>
        <p style="text-align:center; color:#555; font-size:1.05em; margin-top:10px;">
            Check if a news article is real or fake using an SVM-based AI model.
        </p>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="News Text",
                placeholder="Paste or type a news article here...",
                lines=8,
            )
            predict_btn = gr.Button("🚀 Analyze", variant="primary")

        with gr.Column(scale=1):
            result_label = gr.Label(label="Prediction")
            confidence = gr.Label(label="Confidence Level")

    examples = gr.Examples(
        examples=[
            ["The government announced a new plan to improve public education."],
            ["Aliens were spotted landing in the desert according to an anonymous source."],
            ["Scientists discover new species of fish in the Pacific Ocean."],
        ],
        inputs=input_text,
    )

    predict_btn.click(predict_news, inputs=input_text, outputs=[result_label, confidence])

    # Footer
    gr.HTML(
        """
        <hr style="margin-top:40px; margin-bottom:10px; border: none; border-top: 1px solid #ccc;">
        <div style="text-align:center; font-size:1em; color:#333;">
            <b>Developed by Ahmed</b> &nbsp;|&nbsp; <span style="color:#007bff;">SVM Model</span>
        </div>
        """
    )

# -------------------------------
# Launch App
# -------------------------------
if __name__ == "__main__":
    demo.launch()
