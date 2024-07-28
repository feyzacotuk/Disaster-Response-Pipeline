import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Veriyi yükleyin
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Modeli yükleyin
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    # Görselleştirmeler için gerekli verileri çıkarın
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    # Görselleştirmeleri oluşturun
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Mesaj Türlerinin Dağılımı',
                'yaxis': {
                    'title': "Sayı"
                },
                'xaxis': {
                    'title': "Tür"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Mesaj Kategorilerinin Dağılımı',
                'yaxis': {
                    'title': "Sayı"
                },
                'xaxis': {
                    'title': "Kategori"
                }
            }
        }
    ]
    
    # Plotly grafiklerini JSON olarak kodlayın
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # Kullanıcı girdisini query olarak kaydedin
    query = request.args.get('query', '') 

    # Modeli kullanarak query'nin sınıflandırmasını tahmin edin
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
