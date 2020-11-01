import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import os
import sys
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import text, sequence
import pickle


# To load dataset
DATA_URL = ("Data\Original_data.csv")
@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data = data.drop(['tweet_id', 'author'], axis=1)
    return data

data = load_data()


st.title("Sentiment analysis 😎")
st.sidebar.title("Sentiment analysis 😎")

st.markdown("This analysis is a streamlit dashboard to analyze the sentiment")
st.sidebar.markdown("This analysis is a streamlit dashboard to analyze the sentiment")





 
#user text
user_text = st.text_input('WRITE SOME TEXT TO PREDICT ITS SENTIMENT')
sentiments = []


#tokenizing the user_text
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
tokenized = tokenizer.texts_to_sequences([user_text])
x = sequence.pad_sequences(tokenized,maxlen = 20)

#for happy or sad
hap_sad = open('Trials\\Idea\\hap_sad\\arch_happiness_sadness_model.json', 'r')
loaded_hap_sad_model = hap_sad.read()
hap_sad.close()
hap_sad_model = model_from_json(loaded_hap_sad_model)
hap_sad_model.load_weights("Trials\\Idea\\hap_sad\\happiness_sadness_weights.h5")
y_hap_sad = hap_sad_model.predict(x)

if y_hap_sad[0][0]>=0.5:
    sentiments.append('Happy')
if y_hap_sad[0][0]<0.5:
    sentiments.append('Sad') 

#for fun or boredom
fun_bore = open('Trials\Idea\\fun_boredom\\arch_fun_boredom_model.json', 'r')
loaded_fun_bore_model = fun_bore.read()
fun_bore.close()
fun_bore_model = model_from_json(loaded_fun_bore_model)
fun_bore_model.load_weights("Trials\\Idea\\fun_boredom\\fun_boredom_weights.h5")
y_fun_bore = fun_bore_model.predict(x)

if y_fun_bore[0][0]<=0.5:
    sentiments.append('Boredom')
if 'Sad' not in sentiments:    
    if y_fun_bore[0][0]>0.5:
        sentiments.append('Fun') 

#for happy or worry
hap_wor = open('Trials\\Idea\\hap_worry\\arch_happiness_worry_model.json', 'r')
loaded_hap_wor_model = hap_wor.read()
hap_wor.close()
hap_wor_model = model_from_json(loaded_hap_wor_model)
hap_wor_model.load_weights("Trials\Idea\hap_worry\happiness_worry_weights.h5")
y_hap_wor = hap_wor_model.predict(x)

if y_hap_wor[0][0]<0.5:
    sentiments.append('Worry')


#for love or hate
luv_hate = open('Trials\\Idea\\love_hate\\arch_love_hate_model.json', 'r')
loaded_luv_hate_model = luv_hate.read()
luv_hate.close()
luv_hate_model = model_from_json(loaded_luv_hate_model)
luv_hate_model.load_weights("Trials\\Idea\\love_hate\\love_hate_weights.h5")
y_luv_hate = luv_hate_model.predict(x)

if y_luv_hate[0][0]>0.5:
    sentiments.append("Love")
if y_luv_hate[0][0]<=0.5:
    sentiments.append('Hate') 

#for relief or sad
relief_sad = open('Trials\\Idea\\relief_sad\\arch_relief_sadness_model.json', 'r')
loaded_relief_sad_model = relief_sad.read()
relief_sad.close()
relief_sad_model = model_from_json(loaded_relief_sad_model)
relief_sad_model.load_weights("Trials\\Idea\\relief_sad\\relief_sadness_weights.h5")
y_relief_sad = relief_sad_model.predict(x)

if y_relief_sad[0][0]>0.5:
    sentiments.append('Relief')



if len(user_text) == 0:
    
    st.write('')
    sentiments = []
else: 
    st.write(f'The sentiments for "{user_text}"')   
    if 'Happy' in sentiments:
        st.markdown('Happy 😃')
    if 'Sad' in sentiments:
        st.markdown('Sad 😢')    
    if 'Love' in sentiments:
        st.markdown('Love ❤️')
    if 'Hate' in sentiments:
        st.markdown('Hate 💔')
    if 'Fun' in sentiments:
        st.markdown('Fun 🤣')    
    if 'Relief' in sentiments:
        st.markdown('Relief 😌')
    if 'Worry' in sentiments:
        st.markdown('Worry 😟')    
    if 'Boredom' in sentiments:
        st.markdown('Boredom 🥱')





# to pick a random tweet
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio("Sentiment", ('boredom','fun','happiness','hate','love','relief','sadness','worry'))
st.sidebar.markdown(data.query("sentiment == @random_tweet")[["content"]].sample(n=1).iat[0, 0])

# Plotting barplot and piechart
st.sidebar.markdown("### Numbers of tweets by Sentiment")
select = st.sidebar.selectbox("Visualization Type", ["Histogram", "Pie Chart"], key='1')
sentiment_count = data["sentiment"].value_counts()
sentiment_count = pd.DataFrame({"Sentiment": sentiment_count.index, "Tweets": sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Numbers of tweets by Sentiment")
    if select == "Histogram":
        fig = px.bar(sentiment_count, x='Sentiment',
                     y='Tweets', color="Tweets", height=500)
        st.plotly_chart(fig)

    else:
        fig = px.pie(sentiment_count, values="Tweets", names="Sentiment")
        st.plotly_chart(fig)




# Building a wordcloud
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio(
    "Display word for sentiment", ('boredom','fun','happiness','hate','love','relief','sadness','worry'))

if st.sidebar.checkbox("SHOW", False):
    st.header(f"Word cloud for {word_sentiment} sentiment")
    df = data[data['sentiment'] == word_sentiment]
    words = " ".join(df["content"])
    processed_words = " ".join([word for word in words.split(
    ) if 'http' not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black',
                          height=600, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()



