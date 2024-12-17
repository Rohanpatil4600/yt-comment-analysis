# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Expose port
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]
