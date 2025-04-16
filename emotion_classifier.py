from transformers import pipeline

classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")

def get_emotion(text):
    # Perform emotion classification
    results = classifier(text)
    return (results[0]['label'], results[0]['score'])