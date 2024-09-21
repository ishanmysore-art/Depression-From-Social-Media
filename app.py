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
print("RUNNING...")
from keras.models import load_model

# Create an app object using the Flask class.

app = Flask(__name__)

# Load the trained xgboost model and tfidf vectorizer. (Pickle file)

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = load_model('mlpbestmodel.keras')

# Define the route to be home. 
# The decorator below links the relative route of the URL to the function it is decorating.
# Here, home function is with '/', our root directory. 
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder.

# Use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

# You can use the methods argument of the route() decorator to handle different HTTP methods.
# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission. 
# Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():
    acc_type = [x for x in request.form.values()][0].lower()    # either twitter or instagram
    handle = [x for x in request.form.values()][1]      # social media handle

    # For Twitter handle
    if acc_type == 'twitter':
        try:
            client = Client('en-US')    # set up a Twikit client to parse the account
            async def fetch_tweets():
                # loads the login information of a bot Twitter account I made
                client.load_cookies(path='cookies.json')
                user = await client.get_user_by_screen_name(handle)   # gets the user profile given by handle
                tweets_to_store = []    # list to store tweets
                # retrieve 100 tweets from the user
                tweets = await client.get_user_tweets(user_id=user.id, tweet_type='Tweets', count=100)
                # the tweets come in batches of 20
                # so we need to loop through 5 times and add each of the 20 tweets to our list
                for i in range(5):
                    for tweet in tweets:
                        tweets_to_store.append(tweet.full_text)
                    tweets = await tweets.next()
                return tweets_to_store
            X_test = asyncio.run(fetch_tweets())
        except errors.UserNotFound:
            return render_template('error.html', prediction_text='The user was not found. Please go back to the previous page and try again.')
        except errors.TooManyRequests:
            return render_template('error.html', prediction_text='Too many requests. Please try again tomorrow.')

    # For Instagram handle
    else:
        try:
            loader = instaloader.Instaloader()  # set up an Instaloader to parse the account
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
        except instaloader.exceptions.ConnectionException:
            return render_template('error.html')
        except instaloader.exceptions.ProfileNotExistsException:
            return render_template('error.html')
    
    # if someone has posted nothing, we will not classify them as at risk for depression
    # if len(X_test)==0:
    #     return render_template('prediction_no_risk.html')
    
    X_test = pd.Series(np.array(X_test, dtype=object))      # convert the list of social media posts to a pandas series for transforming
    X_test = tfidf.transform(X_test.values.astype('U')).toarray()       # using the tfidf vectorizer, assign weights to words in each of the posts
    y_prob = model.predict(X_test)                    # have the xgboost model predict the probability of depression (using the weights)
    
    # Count number of depressed posts
    length = X_test.shape[0]
    num_depressed = 0
    for val in y_prob:
        if val > 0.5:
            num_depressed += 1

    # Print out number of depressed posts and number of posts the model considered
    print("Num depressed: " + str(num_depressed))
    print("Num posts considered: " + str(length))
    
    # Determine a risk factor based on the percentage of posts classified as depressed
    # >= 80% -- severe risk
    # >= 50% -- moderate risk
    # >= 20% -- mild risk
    # < 20% -- no risk
    if num_depressed >= (0.60 * length):
        template = 'prediction_severe_risk.html'
    elif num_depressed >= (0.40 * length):
        template = 'prediction_moderate_risk.html'
    elif num_depressed >= (0.20 * length):
        template = 'prediction_mild_risk.html'
    else:
        template = 'prediction_no_risk.html'

    # Render a template with the appropriate prediction text
    return render_template(template)

# When the Python interpreter reads a source file, it first defines a few special variables. 
# For now, we care about the __name__ variable.
# If we execute our code in the main program, like in our case here, it assigns
#  __main__ as the name (__name__). 
# So if we want to run our code right here, we can check if __name__ == __main__
# if so, execute it here. 
# If we import this file (module) to another file then __name__ == app (which is the name of this python file).
if __name__ == "__main__":
  app.run()