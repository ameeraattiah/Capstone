import io
import os
import re
import json
import pandas as pd
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files['file']
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        options = request.form.getlist('options[]')
        output_format = request.form.get('format', 'csv')

        # Load file
        if ext == '.csv':
            df = pd.read_csv(file)
        elif ext == '.tsv':
            df = pd.read_csv(file, sep='\t')
        elif ext == '.json':
            df = pd.read_json(file)
        elif ext == '.xlsx':
            df = pd.read_excel(file)
        else:
            return jsonify({'error': f"Unsupported file type: {ext}"}), 400

        # Make sure 'text' column exists before text-based processing
        if 'text' not in df.columns:
            return jsonify({'error': "'text' column is required for processing"}), 400

        # Perform selected cleaning
        if 'deduplication' in options:
            df.drop_duplicates(inplace=True)
        if 'html' in options:
            df['text'] = df['text'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
        if 'noise' in options:
            df['text'] = df['text'].astype(str).str.replace(r'[^\w\s]', '', regex=True)
        if 'normalization' in options:
            df['text'] = df['text'].astype(str).str.replace(r'[^\u0600-\u06FF\s]', '', regex=True)
        if 'tokenization' in options:
            df['text'] = df['text'].astype(str).apply(lambda x: ' '.join(x.split()))
        if 'diacritics' in options:
            df['text'] = df['text'].astype(str).str.replace(r'[\u064B-\u0652]', '', regex=True)
        if 'blocklist' in options:
            blocklist = ['مثال', 'حذف']
            df = df[~df['text'].astype(str).apply(lambda x: any(word in x for word in blocklist))]
        if 'language' in options:
            df = df[df['text'].astype(str).str.contains(r'[\u0600-\u06FF]', regex=True)]

        # Prepare output
        output = io.StringIO()
        filename = f'processed.{output_format}'
        
        if output_format == 'json':
            df.to_json(output, orient='records', force_ascii=False)
            mimetype = 'application/json'
            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        elif output_format == 'tsv':
            df.to_csv(output, sep='\t', index=False)
            mimetype = 'text/tab-separated-values'
            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        elif output_format == 'xlsx':
            output_bytes = io.BytesIO()
            df.to_excel(output_bytes, index=False, engine='openpyxl')
            output_bytes.seek(0)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:  # default: CSV
            df.to_csv(output, index=False)
            mimetype = 'text/csv'
            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        
        return send_file(output_bytes, mimetype=mimetype, as_attachment=True, download_name=filename)


        output.seek(0)


    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
