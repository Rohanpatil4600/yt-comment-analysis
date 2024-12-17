# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import dagshub
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


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
    # dagshub.init(repo_owner='Rohanpatil4600', repo_name='YT_comment', mlflow=True)
    # mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")
    dagshub_token=os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] =dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] =dagshub_token
    dagshub_url = "https://dagshub.com"
    repo_owner= "Rohanpatil4600"
    repo_name= "YT_comment"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    
    # Load the model and vectorizer
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)

    # Debugging: Confirm model and vectorizer are loaded
    print("Model loaded successfully.")
    print(f"Model type: {type(model)}")
    print("Vectorizer loaded successfully.")
    print(f"Vectorizer type: {type(vectorizer)}")

    return model, vectorizer

# Ensure input matches the vectorizer's vocabulary
def align_input_to_vocab(vectorizer, preprocessed_comments):
    """Align input to match the vectorizer's training vocabulary."""
    try:
        input_transformed = vectorizer.transform(preprocessed_comments)
        trained_vocab_size = len(vectorizer.get_feature_names_out())

        # Ensure consistency in shape by padding with zeros
        if input_transformed.shape[1] < trained_vocab_size:
            padding = csr_matrix((input_transformed.shape[0], trained_vocab_size - input_transformed.shape[1]))
            input_transformed = hstack([input_transformed, padding])

        return input_transformed
    except Exception as e:
        print(f"Error during alignment to vocabulary: {e}")
        raise

# Initialize model and vectorizer
model, vectorizer = load_model_and_vectorizer("my_model", "4", "./tfidf_vectorizer.pkl")
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = align_input_to_vocab(vectorizer, preprocessed_comments)
        transformed_comments_dense = transformed_comments.toarray()  # Convert to dense matrix
        # Retrieve feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        print(f"Number of feature names: {len(feature_names)}")  # Log only the count of feature names

        # Create a pandas DataFrame with feature names
        import pandas as pd
        input_df = pd.DataFrame(transformed_comments_dense, columns=feature_names)
        print(f"Input DataFrame shape: {input_df.shape}")  # Log only the DataFrame's shape

        # Make predictions
        predictions = model.predict(input_df).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

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
        transformed_comments_dense = transformed_comments.toarray()  # Convert to dense matrix

        # Retrieve feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        print(f"Number of feature names: {len(feature_names)}")  # Log only the count of feature names

        # Create a pandas DataFrame with feature names
        import pandas as pd
        input_df = pd.DataFrame(transformed_comments_dense, columns=feature_names)
        print(f"Input DataFrame shape: {input_df.shape}")  # Log only the DataFrame's shape

        # Make predictions
        predictions = model.predict(input_df).tolist()

        # Return response
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(response)

    except Exception as e:
        print("Error Traceback:")
        print(traceback.format_exc())  # Print detailed error traceback
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
