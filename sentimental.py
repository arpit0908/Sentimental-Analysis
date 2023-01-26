#Description : Twitter sentimental Analysis which analyses the sentiments of the Tweets using Python. 

# Import the Libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#style is used(dark backround, ggstyles, 538 etc.)
plt.style.use('fivethirtyeight')


# load the data
from google.colab import files
uploaded = files.upload()

#get the data/read the data
log = pd.read_csv('credentials.csv')

#Twitter API credentials
api_key = log['Keys'][0]
api_key_secret = log['Keys'][1]
access_token = log['Keys'][2]
access_token_secret = log['Keys'][3]

# Create the authentication object
authenticate = tweepy.OAuthHandler(api_key,api_key_secret)

# Set the access token and token secret
authenticate.set_access_token(access_token,access_token_secret) 

#Create the API object while passing in the authentication information
api =  tweepy.API(authenticate,wait_on_rate_limit = True)

# Extract 100 tweets from the twitter users
posts = api.user_timeline(screen_name = "billgates",count= 100,lang = "en", tweet_mode = "extended")

# Print the last 5 tweets from the account
print("Show the 5 recent tweets: \n")
i=1 
for tweet in posts[0:5]:
  print(str(i) + ') '+ tweet.full_text + '\n')
  i=i+1

#create a dataframe with a column called Tweets
df = pd.DataFrame( [tweet.full_text for tweet in posts], columns=['Tweets'])

#show the first 5 rows of data
df.head()

#Pre-Processing the text

#function to clean the tweets

def cleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9_]+', '', text) #Removes @mentions
  text = re.sub(r'#', '', text) # Removing the '#' Symbol
  text = re.sub(r'RT : ', '', text) #Removing RT
  text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', text)#Remove the hyper link

  return text

df['Tweets']= df['Tweets'].apply(cleanTxt) 

# create a function to get the subjectivity 
def  getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# create a function to get the polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

#create two new columns
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity']     = df['Tweets'].apply(getPolarity)

df

#visualizing the words using word cloud
allWords = ' '.join( [twts for twts in df['Tweets']] )
wordCloud = WordCloud(width= 500,height=300,random_state=21,max_font_size =119).generate(allWords)

plt.imshow(wordCloud,interpolation = "bilinear")
plt.axis('off')
plt.show()

#Create a function to compute the negative,neutral and positive analysis
def getAnalysis(score):
  if score<0:
     return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
  
  df['Analysis'] = df['Polarity'].apply(getAnalysis)

  df
  
  # Plot the Polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjevtivity')
plt.show()

# Get the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

round( (ptweets.shape[0]/df.shape[0])*100,1)


#Get the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']

round((ntweets.shape[0]/df.shape[0])*100,1)


#Get the percentage of neutral tweets
nutweets = df[df.Analysis == 'Neutral']
nutweets = nutweets['Tweets']

round((nutweets.shape[0]/df.shape[0])*100,1)


#show the value counts

df['Analysis'].value_counts()

#plot and visualize the counts
plt.title('Sentimental Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()


