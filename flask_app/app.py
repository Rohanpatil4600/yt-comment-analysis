from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import dagshub

app = Flask(__name__)
CORS(app)

# Preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split() if word not in stop_words])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load the model and vectorizer."""
    dagshub.init(repo_owner='Rohanpatil4600', repo_name='YT_comment', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Ensure input matches the vectorizer's vocabulary
def align_input_to_vocab(vectorizer, preprocessed_comments):
    """Align input to the vectorizer's vocabulary."""
    input_transformed = vectorizer.transform(preprocessed_comments)
    vocab_size = len(vectorizer.get_feature_names_out())
    
    # If the input shape has fewer columns, pad it with zeros
    if input_transformed.shape[1] < vocab_size:
        padding = csr_matrix((input_transformed.shape[0], vocab_size - input_transformed.shape[1]))
        input_transformed = csr_matrix(np.hstack((input_transformed.toarray(), padding.toarray())))
    
    return input_transformed

# Initialize model and vectorizer
model, vectorizer = load_model_and_vectorizer("my_model", "4", "./tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Align input to the vectorizer's vocabulary
        transformed_comments = align_input_to_vocab(vectorizer, preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()

        # Return response
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
