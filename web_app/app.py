# simple_app.py - Simplified Flask Backend

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import re
import os

# Try to import NLTK components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
    
except ImportError:
    print("⚠️ NLTK not available. Using basic text cleaning.")
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    NLTK_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Global variables for model components
model = None
title_vectorizer = None
text_vectorizer = None
label_encoder = None

def load_model_components():
    """Load all model components - automatically finds files in same folder"""
    global model, title_vectorizer, text_vectorizer, label_encoder
    
    try:
        # Get current directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔍 Looking for model files in: {current_dir}")
        
        # List all files in current directory
        print(f"📁 Files found:")
        all_files = [f for f in os.listdir(current_dir) if f.endswith('.joblib')]
        for file in all_files:
            print(f"   • {file}")
        
        if not all_files:
            print("❌ No .joblib files found!")
            print("📋 Please put your model files in the same folder as this script")
            return False
        
        # Find model files automatically
        model_file = None
        vectorizers_file = None
        encoder_file = None
        
        for file in all_files:
            if 'model' in file and 'vectorizers' not in file and 'encoder' not in file:
                model_file = file
            elif 'vectorizers' in file:
                vectorizers_file = file
            elif 'encoder' in file:
                encoder_file = file
        
        # Check if all files found
        missing_files = []
        if not model_file:
            missing_files.append("model file (should contain 'model' in name)")
        if not vectorizers_file:
            missing_files.append("vectorizers file (should contain 'vectorizers' in name)")
        if not encoder_file:
            missing_files.append("encoder file (should contain 'encoder' in name)")
        
        if missing_files:
            print("❌ Missing files:")
            for missing in missing_files:
                print(f"   • {missing}")
            return False
        
        # Load the model components
        print(f"📊 Loading model: {model_file}")
        model = joblib.load(os.path.join(current_dir, model_file))
        
        print(f"📊 Loading vectorizers: {vectorizers_file}")
        vectorizers = joblib.load(os.path.join(current_dir, vectorizers_file))
        
        print(f"📊 Loading encoder: {encoder_file}")
        label_encoder = joblib.load(os.path.join(current_dir, encoder_file))
        
        title_vectorizer = vectorizers['title_vectorizer']
        text_vectorizer = vectorizers['text_vectorizer']
        
        print("✅ All model components loaded successfully!")
        print(f"📈 Model type: {type(model).__name__}")
        print(f"🏷️ Available subject categories: {list(label_encoder.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Make sure all .joblib files are in the same folder as this script")
        print("   2. Check that files downloaded correctly from Kaggle")
        print("   3. Verify file names contain 'model', 'vectorizers', or 'encoder'")
        return False

def clean_text(text):
    """Clean text using basic preprocessing"""
    if not text:
        return ""
    
    # Convert to string
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Remove numbers 
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # 6. Lemmatize (if NLTK available)
    if NLTK_AVAILABLE:
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
        
    return text

def prepare_features(title, text, subject):
    """Prepare features exactly like your training pipeline"""
    try:
        # Clean the text
        title_clean = clean_text(title)
        text_clean = clean_text(text)
        
        # Transform using TF-IDF vectorizers
        title_tfidf = title_vectorizer.transform([title_clean])
        text_tfidf = text_vectorizer.transform([text_clean])
        
        # Encode subject
        subject_encoded = label_encoder.transform([subject])
        
        # Combine features
        title_features = title_tfidf.toarray()
        text_features = text_tfidf.toarray()
        subject_features = subject_encoded.reshape(-1, 1)
        
        combined_features = np.hstack([title_features, text_features, subject_features])
        
        return combined_features
        
    except Exception as e:
        raise Exception(f"Feature preparation failed: {str(e)}")

@app.route('/')
def home():
    """Serve a simple test page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detector API</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: 0 auto; }
            h1 { color: #333; }
            .status { padding: 15px; border-radius: 5px; margin: 10px 0; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Fake News Detector API</h1>
            <div class="status success">
                ✅ Flask server is running!
            </div>
            <p>Your API endpoints:</p>
            <ul>
                <li><strong>/</strong> - This page</li>
                <li><strong>/predict</strong> - Make predictions (POST)</li>
                <li><strong>/health</strong> - Health check</li>
            </ul>
            <p>To use the web interface:</p>
            <ol>
                <li>Open your <code>index.html</code> file in a web browser</li>
                <li>Make sure this Flask server is running</li>
                <li>Test your fake news detector!</li>
            </ol>
            <button onclick="window.location.href='/health'">Check API Health</button>
        </div>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make fake news prediction"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()
        subject = data.get('subject', '').strip()
        
        # Validate input
        if not title or not text or not subject:
            return jsonify({'error': 'Title, text, and subject are required'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Prepare features
        features = prepare_features(title, text, subject)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'label': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': float(max(probabilities)),
            'fake_probability': float(probabilities[0]),
            'real_probability': float(probabilities[1])
        }
        
        # Add CORS header
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        error_response = jsonify({'error': f'Prediction failed: {str(e)}'})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'nltk_available': NLTK_AVAILABLE
    })

if __name__ == '__main__':
    print("🚀 Starting Fake News Detection API...")
    print("📋 Required packages: flask, joblib, scikit-learn, numpy")
    print("📋 Optional packages: nltk (for better text cleaning)")
    
    # Load model components
    if load_model_components():
        print("🌐 Starting Flask server on http://localhost:5000")
        print("💡 Open your index.html file in a browser to use the interface")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check your model files.")
        print("📝 Make sure these files exist in the same folder:")
        print("   - fake_news_model_TIMESTAMP.joblib")
        print("   - fake_news_vectorizers_TIMESTAMP.joblib") 
        print("   - fake_news_encoder_TIMESTAMP.joblib")