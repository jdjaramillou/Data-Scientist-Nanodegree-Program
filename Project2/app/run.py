import json
import plotly
import pandas as pd
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
import joblib
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

def number_messages_genre(df):
    """Create a bar plot (interactive ploty format) per category
    Args:
    df: SQL loaded Dataframe
    Returns: bar plot of categories and top 10 of more important categories
    """

    # Sort by categories descending
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    data = [Bar(
        x=genre_names,
        y=genre_counts
    )]

    layout = Layout(
        title="# Messages per Genre",
        xaxis=dict(title='Genre'),
        yaxis=dict(title='#Messages')
    )

    return Figure(data=data, layout=layout)



def number_messages(df):
    """Create a bar plot (interactive ploty format) per category
    Args:
    df: SQL loaded Dataframe
    Returns: bar plot of categories and top 10 of more important categories
    """

    # Sort by categories descending
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False)

    data = [Bar(
        x=categories.index,
        y=categories
    )]

    layout = Layout(
        title="# Messages per Category",
        xaxis=dict(title='Category'),
        yaxis=dict(title='# Messages'
        )
    )

    return Figure(data=data, layout=layout), categories.index[:10]


def categories_per_genre(df, top_categories):
    """Create a stacked barplot per gender to the to 10 categories
    Args:
    df: SQL loaded Dataframe
    top_categories: top 10 more repeated categories
    Returns: plot of categories by gener
    """

    # Group by categories
    genres = df.groupby('genre').sum()[top_categories]

    data = []
    for cat in genres.columns[1:]:
        data.append(Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = Layout(
        title="Top 10 Categories per genre",
        xaxis=dict(
            title='Genres',
            tickangle=45
        ),
        yaxis=dict(
            title='# messages per category',
        )
    )

    return Figure(data=data, layout=layout)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # encode plotly graphs in JSON
    graphs = [plot1, plot2, plot3]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print([query]) 
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")

# create plots
plot1 = number_messages_genre(df)
plot2, top_cat = number_messages(df)
plot3 = categories_per_genre(df, top_cat)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()