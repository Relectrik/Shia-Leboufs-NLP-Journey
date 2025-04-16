from transformers import pipeline


def get_emotion(text):
    classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")

    # Perform emotion classification
    results = classifier(text)
        
    return (results[0]['label'], results[0]['score'])