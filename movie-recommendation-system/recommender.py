import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

df = pd.read_csv("https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv")

features = ['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string
df["combined_features"] = df.apply(combine_features,axis=1)

cv = CountVectorizer() #creating new CountVectorizer() object
count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


def get_recommendations(title):
    movie_user_likes = title
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index])) 
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    i=1
    print("Top 10 similar movies to "+movie_user_likes+" are:\n")
    tit=[]
    year=[]
    for element in sorted_similar_movies:
        tit.append(get_title_from_index(element[0]))
        year.append(pd.DatetimeIndex(df[df.index == element[0]]['release_date']).year[0])
        i=i+1
        if i>10:
            break
    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = year
    return return_df

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        if(m_name in df['title'].values):
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
            return flask.render_template('positive.html',movie_names=names,movie_date=dates,search_name=m_name)
        else:
            return(flask.render_template('negative.html',name=m_name))
            

if __name__ == '__main__':
    app.run()