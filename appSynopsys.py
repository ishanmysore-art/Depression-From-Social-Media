'''
Application that predicts if an adolescent is at risk for depression by examining their post data from Twitter or Instagram.
Trained on 350,000+ lines of data containing depressed and non-depressed posts from users across various social media sites.

The user will input their handle from a public Twitter or Instagram account. Then, the application will predict 4 levels of 
risk for depression: severe risk, moderate risk, mild risk, and no risk.

If the application predicts severe or moderate risk for depression, it will recommend the adolescent to seek an actual diagnosis
from a mental health professional. If the application predicts mild risk, it will guide the adolescent to some links for information
and practices that may help mitigate symptoms.

@author Ishan Mysore
@version 7/30/2024

Citations:
    Barton, D. (2024, March 18). How to scrape Twitter data using python without using Twitter's API. Apify Blog. https://blog.apify.com/how-to-scrape-tweets-and-more-on-twitter-59330e6fb522/ 
    Bhattiprolu, S. (2022, May 4). 268 - How to deploy your trained machine learning model into a local web application? [Video]. YouTube. https://www.youtube.com/watch?v=bluclMxiUkA
    Pawar, A. (2023, January 27). Introduction to Instaloader module in Python. GeeksforGeeks. https://www.geeksforgeeks.org/introduction-to-instaloader-module-in-python/
'''


# Importing packages

print("IMPORTING...")
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from twikit import Client
from twikit import errors
import asyncio
import instaloader
import pytesseract
from PIL import Image
import os
import shutil
import requests
from io import BytesIO
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import string
from bs4 import BeautifulSoup
import emoji
from nltk.stem import WordNetLemmatizer
print("RUNNING...")
from keras.models import load_model



# Load the trained model and tfidf vectorizer. (Pickle file)

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = load_model('mlpbestmodel.keras')


def predictTwitter(handle):
    '''Retrieve 100 most recent posts from given twitter account'''
    try:
        # set up a Twikit client to parse the account
        client = Client('en-US')   

        async def fetch_tweets():
            # load information from client twitter account
            client.load_cookies("cookies.json")
            # gets the user profile given by handle
            user = await client.get_user_by_screen_name(handle)
            tweets_to_store = []    # list to store tweets
            # retrieve 100 tweets from the user
            tweets = await client.get_user_tweets(user_id=user.id, 
                                tweet_type='Tweets', count=100)
            # the tweets come in batches of 20
            # so we need to loop through 5 times
            # add each of the 20 tweets to our list
            for i in range(5):
                for tweet in tweets:
                    tweets_to_store.append(tweet.full_text)
                tweets = await tweets.next()
            return tweets_to_store
        
        # store the list of tweets for model prediction
        X_test = asyncio.run(fetch_tweets())
        return X_test
    except errors.UserNotFound:
        return 'The user was not found.' \
        'Please go back to the previous page and try again.'
    except errors.TooManyRequests:
        return 'Too many requests. Please try again tomorrow.'

# For Instagram handle
def predictInsta(handle):
    try:
        loader = instaloader.Instaloader()
        filepath = r'/Users/ishanmysore/Documents/DepressionWebDevelopment/imysore-depressionprediction/instagram_session'
        # Load session (only if you've previously logged in)
        try:
            loader.load_session_from_file('instagram_session', filepath)
        except FileNotFoundError:
            # If no session file, login once
            loader.context.log("Login session not found, logging in...")
            loader.login('cooldude23571113', '23571113')
            loader.save_session_to_file('instagram_session',filepath)
        profile = instaloader.Profile.from_username(loader.context, handle) # get the user profile
        count = 0       # count of posts
        X_test = []     # list to store posts
        image_urls = [] # list to store image urls to read later
        # go through the 100 most recent Instagram posts
        # save each image url to the list of image urls
        # save each post caption to the list
        for post in profile.get_posts():
            count += 1
            image_urls.append(post.url)
            if not (post.caption is None) and len(post.caption) > 0:
                X_test.append(post.caption)
            if count >= 100:
                break
        print("All posts have been downloaded.")
        # loop through the list of saved urls
        # get the image given by the url
        # read any text contained in each image
        # save this text to the list
        for url in image_urls:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            text = pytesseract.image_to_string(image)
            X_test.append(text)
        return X_test
    except instaloader.exceptions.ConnectionException:
        return "An error was encountered."
    except instaloader.exceptions.ProfileNotExistsException:
        return "That profile does not exist."

# if someone has posted nothing, we will not classify them as at risk for depression

# Dictionary of common short forms for words
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_urls(text):
  return re.sub(r'http\S+|www\S+', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def replace_chat_words(text):
  words = text.split()
  for i, word in enumerate(words):
    if word.lower() in chat_words:
      words[i] = chat_words[word.lower()]
  return ' '.join(words)

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = text.split()
  filtered_words = [word for word in words if word.lower() not in stop_words]
  return ' '.join(filtered_words)

def remove_emojis(text):
  return emoji.demojize(text)

list_of_accounts = ['depressingmsgs', 'Areallifeloserr', 'thisusertwtss']

X_test = predictTwitter('depressingmsgs')
X_test = pd.Series(np.array(X_test, dtype=object))      # convert the list of social media posts to a pandas series for transforming
X_test_orig = X_test # for later

print("PREPROCESSING...")
X_test = X_test[X_test.str.strip().astype(bool)]
# Converts all characters to lowercase
X_test = X_test.str.lower()
# Remove HTML tags
X_test = X_test.apply(remove_html_tags)
# Removes all URLs
X_test = X_test.apply(remove_urls)
# Removes all punctuation
X_test = X_test.apply(remove_punctuation)
# Replace chat words with their full forms
X_test = X_test.apply(replace_chat_words)
# Removes all stopwords
X_test = X_test.apply(remove_stopwords)
# Replaces emojis with their textual representation
X_test = X_test.apply(remove_emojis)
# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
X_test = X_test.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word , pos='v') for word in x.split()]))
# Gets rid of empty rows once again (in case preprocessing led to an empty row)
X_test = X_test[X_test.str.strip().astype(bool)]

print("TRANSFORMING...")
X_test_matrix = tfidf.transform(X_test.values.astype('U')).toarray()       # using the tfidf vectorizer, assign weights to words in each of the posts
y_prob = model.predict(X_test_matrix)                    # have the model predict the probability of depression (using the weights)


# Get feature names from the vectorizer
feature_names = np.array(tfidf.get_feature_names_out())

def get_top_words_for_row(row_idx, tfidf_matrix, feature_names, top_n=10):
    """Get the top N most important words in a given row of the TF-IDF matrix."""
    row_data = tfidf_matrix[row_idx]  # Extract the row
    # Get indices of words with nonzero TF-IDF values
    word_indices = np.where(row_data > 0)[0]
    # Get corresponding TF-IDF scores
    word_tfidf_values = row_data[word_indices]
    # Sort words by importance (highest TF-IDF first)
    top_indices = word_indices[np.argsort(word_tfidf_values)[-top_n:][::-1]]
    # Get feature names corresponding to the top indices
    top_words = feature_names[top_indices]
    return list(top_words)


from prettytable import PrettyTable

table = PrettyTable(["Number", "Original Post", "Preprocessed Post", "Flagged Words", "Model Confidence"])
num_depressed = 0
length = X_test_matrix.shape[0]

np.set_printoptions(threshold=np.inf)
for i in range(len(y_prob)):
    if y_prob[i]>0.5:
        num_depressed += 1
        table.add_row([str(i), X_test_orig[i][0:50], X_test[i][0:50], get_top_words_for_row(i, X_test_matrix, feature_names, top_n=3), str(y_prob[i])])
        # print(str(i), "\t", X_test[i][0:50], "\t", y_prob[i], "\t", get_top_words_for_row(i, X_test_matrix, feature_names, top_n=3))

# Print out number of depressed posts and number of posts the model considered

print(table)

print("\nNum depressed: " + str(num_depressed))
print("Num posts considered: " + str(length))


# # Get the feature names from the TF-IDF vectorizer
# for i in range(len(y_prob)):
#     feature_names = np.array(tfidf.get_feature_names_out())
#     # Get the absolute sum of weights across all neurons in the first layer
#     word_importance = np.abs(model.layers[0].get_weights()[0]).sum(axis=1)
#     # Get indices of top 20 most important words
#     top_indices = np.argsort(word_importance)[-5:]
#     # Print the most important words for predicting depression
#     print("Top 5 words associated with depression risk in post" + str(i))
#     for j in reversed(top_indices):
#         print(f"{feature_names[j]}: {word_importance[j]}")
