from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModel


app= Flask(__name__)

#Load the Hugging Face Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

@app.route("/predict", methods=["POST"])
def predict():
    #Get the text from the request
    text = request.json["text"]

    #Encode the text
    encoded_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

    #Get the model's predictions
    with torch.no_grad():
        predictions = model(**encoded_text)
    
    # Convert the predictions to JSON format
    predictions_json = {
        "label": predictions[0].argmax().item(),
        "score":predictions[0][predictions[0].argmax().item()]
    }

    # Return the JSON predictions
    return jsonify(predictions_json)

    if __name__ == "__main__":
        app.run(debug=True)