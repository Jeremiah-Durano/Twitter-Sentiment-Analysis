import pandas as pd
import numpy as np
import re
import neattext as nt
import neattext.functions as nfx
from textblob import TextBlob

#Load Dataset 
def load_data():
    #Reads in CSV file and puts into a dataframe
    data = pd.read_csv("tweets.csv") 
    return data

class TweetAnalyzer():
    #Creating a function to clean the tweets
    def cleanTweet(self,tweet):
        tweet = re.sub(r'@[A-Za-z0-9]+','',tweet) # Removes @Mentions
        tweet = re.sub(r'#','',tweet) # Removing the '#' symbol
        tweet = re.sub(r'RT[\s]+','', tweet) # Removing RT
        tweet = re.sub(r'https?:\/\/\S+','',tweet) # Removes the hyper link
        tweet = re.sub(r':','',tweet) #Removes ':' 
        tweet = re.sub(r'-','',tweet) #Removes '-' 

        return tweet

    #Function for sentiment Analysis
    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.cleanTweet(tweet))
            
        if analysis.sentiment.polarity > 0: #Positive 
            return 'Positive'
        elif analysis.sentiment.polarity == 0: #Neutral
            return 'Neutral'
        else:
            return 'Negative' #Negative

tweet_df = load_data()
tweet_df = tweet_df[['tweet_text']]
#Clean the tweets column and get it's sentiment 
tweet_analyzer = TweetAnalyzer()
tweet_df['tweet_text'] = tweet_df['tweet_text'].apply(tweet_analyzer.cleanTweet)
tweet_df['sentiment'] =  np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in tweet_df['tweet_text']])
tweet_df.to_csv(r'sentiment_includedv3.csv', index = False, header=True)

#Data Verification
print('Dataset size:' , tweet_df.shape)
print('Columns are:', tweet_df.columns)
tweet_df.info()
#Prints Dataset
#print(tweet_df) 

'''
Counts how many times sentiments occur 1 = Positive 0 = Neutral -1 = Negative
'''
print(pd.value_counts(tweet_df['sentiment']))
