#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:44:09 2018

@author: marta

Minerando os dados do tweet
"""

#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
from time import sleep
import csv
import re
import os
import wget
from pathlib import Path
from datetime import datetime
from datetime import timedelta
account="mattyglesias"
windows_year=5

date_tweet_ini=  datetime.strptime("2014-01-01", "%Y-%m-%d")
date_tweet_end = date_tweet_ini + timedelta(days=365*windows_year) #2018-12-31


      
        
#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""
filepath_download="/home/marta/meu/projfinal/dataset_tweet_en/jpg"

def validchar(txt):
    txt = txt.decode('utf-8')
    
    #remove URL
    element = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', txt, flags=re.MULTILINE)
    
    
   
    #check special characteres
    pat=r'([^a-zA-Z\d\s?":çáàãâéèêẽëíîóôûúü#.!,@_-…-)/)();+*$])' # except => \d - numeric class, \s - whitespace ,a-zA-Z - matches all the letters
    element = re.sub( pat, '', element.lower() ,flags=re.MULTILINE) 
    return element.lower().strip()

def filecsv(alltweets):
     #transform the tweepy tweets into a 2D array that will populate the csv	
    tweets=[]
    for tweet in alltweets:
        media = tweet.entities.get('media', [])        
        d1=tweet.created_at
        
        if (date_tweet_ini <= d1 <= date_tweet_end):   
            if len(media)!=0:
                txt=validchar(tweet.text.encode("utf-8"))
                if media[0]['type']=='photo' and len(txt)>10:
                        
                    img = media[0]['media_url']        
                    print("download tweet:" + str(tweet.id_str))                   
                    
                
                    tweets.append( [tweet.id_str, str(tweet.created_at), txt , str(tweet.retweet_count), str(tweet.favorite_count),img,tweet.author.screen_name] )
        	
                    url = media[0]['media_url']
                    filename = url.split("/")[-1]   
                    my_file = Path(filepath_download + '/' + filename)
                    if not my_file.exists():
                        wget.download(media[0]['media_url'],out=filepath_download)
                        sleep(1)                                 
                        os.rename(filepath_download + '/' + filename, filepath_download + '/' + account + '_' + str(tweet.id) + '.'+ filename.split(".")[1])
                    
                        
        
    #write the csv	
    print("writing the csv file")
    with open('tweets_' + account + '.csv', 'w') as f:
        writer = csv.writer(f)        
        writer.writerow(["id","created_at","text","retweetcount","favoritecount", "img", "author"])
        writer.writerows(tweets)
    
def get_all_tweets(screen_name, alltweets=[], max_id=0,maxreg=20000):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    #authorize twitter, initialize tweepy
    consumer_key = 'xchPewahorbyhcBqw0OTRLA86'
    consumer_secret = 'MDNWvZ7JfpOUMskjvPzdY7mre94wZd7z7oNOJQBjE2ZYMIBx7f'
    access_token = '105881289-4iKWpkslet9oafHc1o2QNmAtWeQnaUJXHgfjk7NG'
    access_secret = 'e2QoL5qKxXFrdg0wRoEoWEHT2cvMLuFGbmKchV8FXzsZX'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    if (not api):
        print ("****Can't Authenticate****")

    #make initial request for most recent tweets (200 is the maximum allowed count)
    if max_id is 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=50)
    else:
        # new new_tweets
        new_tweets = api.user_timeline(screen_name=screen_name, count= 50, max_id=max_id)
    
    if len(alltweets)>=maxreg:
        return alltweets
        
    if len(new_tweets) > 0:
        
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
       	
        # security
        sleep(5)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print("...%s tweets downloaded so far" % (len(alltweets)))
        
        return get_all_tweets(screen_name=screen_name, alltweets=alltweets, max_id=oldest)
    
    
   
    #final tweets
    return alltweets
if __name__ == '__main__':
    #pass in the username of the account you want to download
    tweets=get_all_tweets(account, [], 0)
    print("printing csv file.")
    filecsv(tweets)
    
    
    
"""
reuters,cnn,bbc,bloomberg,wsj,breakingnews,bbcbreaking,cnnbrk,nytimes   time,ABCNewsLive
jayrosen_nyu,mattyglesias       ,mikiebarb,KFILE,pbump

https://www.businessinsider.com/top-20-influential-media-twitter-2010-9-3#13-wall-street-journal-8
https://www.adweek.com/digital/twitter-breaking-news/

https://www.bustle.com/articles/200746-17-journalists-to-follow-on-twitter-in-2017-as-we-enter-a-politically-charged-year-for

DATA- https://stackoverflow.com/questions/49731259/tweepy-get-tweets-among-two-dates
url = media[0]['media_url']
filename = url.split("/")[-1]   
my_file = Path(filepath_download + '/' + filename)
if not my_file.exists():
wget.download(media[0]['media_url'],out=filepath_download)
time.sleep(1)                                 
os.rename(filepath_download + '/' + filename, filepath_download + '/' + str(status.id)+ '.'+ filename.split(".")[1])
"""