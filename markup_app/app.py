import pandas as pd
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime

app = Flask(__name__)

INPUT_CSV = 'data/sample_100.csv'
OUTPUT_CSV = 'data/markup_data.csv'

def load_data():
    if os.path.exists(INPUT_CSV):
        return pd.read_csv(INPUT_CSV, index_col="id")
    return pd.DataFrame()

def load_labeled_data():
    if os.path.exists(OUTPUT_CSV):
        return pd.read_csv(OUTPUT_CSV, index_col="id")
    return pd.DataFrame()

def save_labeled_data(df):
    df.to_csv(OUTPUT_CSV)


@app.route('/')
def index():
    sample_df = load_data()
    labeled_df = load_labeled_data()
    
    if sample_df.empty:
        return render_template('index.html', 
                             no_data=True,
                             completed=True,
                             labeled_count=len(labeled_df))
    
    samples =[]
    for i,s in sample_df.iterrows():
        samples.append({
            "command": s["command"],
            "reasoning": s["reasoning"],
            "description": s["description"],
            "is_command": s["is_command"],
            "index": i
        })
    return render_template('index.html', 
                         samples=samples,
                         no_data=False,
                         labeled_count=len(labeled_df))

@app.route('/save_label', methods=['POST'])
def save_label():
    data = request.get_json()
    
    df = load_data()
    original_row = df.loc[data['index']].to_dict()
    
    labeled_row = {
        **original_row,
        'markup_category': data['category'],
        'markup_reason': data['reason'],
        'real_is_command': data['real_is_command'],
        'markup_timestamp': datetime.now().isoformat(),
        "id": data["index"]
    }
    
    labeled_df = load_labeled_data()
    if labeled_df.empty:
        labeled_df = pd.DataFrame([labeled_row])
        labeled_df.set_index("id", inplace=True)
    else:
        labeled_df = labeled_df[labeled_df.index != data['index']]
        labeled_df = pd.concat([labeled_df, pd.DataFrame([labeled_row]).set_index("id")], axis=0)
    
    save_labeled_data(labeled_df)
    
    return jsonify({'status': 'success'})

@app.route('/get_labeled_data')
def get_labeled_data():
    labeled_df = load_labeled_data()
    return jsonify(labeled_df.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)