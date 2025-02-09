from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from werkzeug.utils import secure_filename
import PyPDF2
import os

app = Flask(__name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {'pdf'}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file):
    """Extract text from PDF file using PyPDF2"""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and extract text"""
    name = request.form.get('name')
    file = request.files.get('file')

    if not name or not file:
        return jsonify({"error": "No Name or file uploaded"}), 400

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_folder = os.path.join(UPLOAD_FOLDER, name)

        # Create user folder if it doesn't exist
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        filepath = os.path.join(user_folder, filename)  # Corrected file path
        file.save(filepath)

        # Extract text from PDF
        extracted_text = extract_text(filepath)

        return jsonify({"extracted_text": extracted_text})

    return jsonify({"error": "Invalid file format. Only PDF allowed"}), 400

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    text = request.json['text']
    max_length = request.json.get('max_length', 100)

    summary = summarizer(text, max_length=max_length, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)
