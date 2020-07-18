#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np

# In[4]:

# Create a panda table with three colums
billboard_data1= pd.DataFrame(columns=['Title', 'Artist', 'Year'])

# In[5]:
# List of years for which hit songs lists will be obtained
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,2014, 2015, 2016, 2017, 2018]
# Appending year to url_template we get the hit song data from wikipedia
url_template = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'

# In[15]:

url = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_2000'
temp_df = pd.read_html(url, header=0)[0]

# In[16]:


temp_df.head()


# In[18]:


for year in years:
    url = url_template + str(year)
    temp_df = pd.read_html(url, header=0)[0]
    temp_df.columns.values[2] = "Artist"
    temp_df = temp_df.drop(temp_df.columns[0], axis=1)
    temp_df['Artist']=temp_df['Artist'].astype(str)
    temp_df.Artist = [artist.split(' featuring')[0] for artist in temp_df.Artist]
    temp_df.Title = [title.strip('\"') for title in temp_df.Title]
    temp_df['Year'] = year
    billboard_data1 = billboard_data1.append(temp_df)


# In[24]:


billboard_data1.head()


# In[23]:


billboard_data1 = billboard_data1.reset_index(drop=True)
billboard_data1['Year'] = pd.to_numeric(billboard_data1['Year'])
billboard_data1['Title'] = billboard_data1['Title'].astype(str)
billboard_data1['Artist'] = billboard_data1['Artist'].astype(str)


# In[26]:


billboard_data1.to_csv('../data-files/billboard_data1.csv')


# In[64]:


billboard_data1=pd.read_csv('../data-files/billboard_data1.csv',index_col=0)
billboard_data1=billboard_data1.drop('Year',axis=1)
billboard_data1 = billboard_data1.drop_duplicates(subset="Title", keep="first")


# In[65]:


#Data integration
spotify_songs = pd.read_csv('../data-files/song_info.csv')


# In[66]:


spotify_songs.head()


# In[67]:


spotify_songs = spotify_songs.drop(['album_names','playlist'], axis=1)


# In[68]:


spotify_songs.head()


# In[69]:


spotify_songs.columns.values


# In[70]:


spotify_songs = spotify_songs.rename(columns={"song_name": "Title", "artist_name": "Artist"})


# In[71]:


spotify_songs.head()


# In[72]:


spotify_songs = spotify_songs.drop_duplicates(subset="Title", keep="first")


# In[73]:


spotify_songs['Top100'] = 0
billboard_data1['Top100'] = 1


# In[74]:


combined_df = pd.concat([spotify_songs, billboard_data1], ignore_index=True, sort=True)


# In[75]:


combined_df = combined_df.drop_duplicates(subset="Title", keep="last")


# In[76]:


combined_df.head()


# In[77]:


combined_df.tail()


# In[78]:


billboard_data1.head()


# In[80]:


combined_df.to_csv('../data-files/songs-list.csv')


# In[81]:


#Getting songs uris
combined_df.shape


# In[1]:


import spotipy
import spotipy.util as util
from tqdm import tqdm_notebook


# In[105]:


def get_spotify_uri(title, artist):
    query = title + " " + artist
    search = sp.search(q=query, limit=50, offset=0, type='track', market='US')
    search_items = search['tracks']['items']
    for i in range(len(search_items)):
            uri = search_items[i]['id']
            return uri
    return 0


# In[106]:


token = util.oauth2.SpotifyClientCredentials(client_id='b7fc404a069d4a0f83049834a7d09a25', client_secret='990a5a3792e9469fac00c288ec610d7d')
cache_token = token.get_access_token()
spotify = spotipy.Spotify(cache_token)
sp = spotipy.Spotify(auth=cache_token)


# In[112]:


titles = list(combined_df.Title)
artists = list(combined_df.Artist)
top100 = list(combined_df.Top100)
spotify_uri = list()
errors = list()


# In[114]:


temp1 = list()


# In[115]:


for i in tqdm_notebook(range(0,13993)):
    uri = get_spotify_uri(titles[i], artists[i])
    
    if uri != 0:
        temp1.append(uri)
    else:
        temp1.append(uri)
        errors.append(i)
spotify_uri = spotify_uri + temp1


# In[123]:


len(spotify_uri)


# In[124]:


songs_with_id = pd.DataFrame()


# In[126]:


songs_with_id['Title'] = titles;
songs_with_id['Artist'] = artists;
songs_with_id['Top100'] = top100;
songs_with_id['URI'] = spotify_uri;
songs_with_id=songs_with_id[songs_with_id.URI!=0]


# In[127]:


songs_with_id.head()


# In[128]:


songs_with_id.tail()


# In[131]:


songs_with_id.shape


# In[133]:


songs_with_id.reset_index(drop=True,inplace=True)


# In[134]:


songs_with_id.tail()


# In[135]:


songs_with_id.to_csv('../data-files/songs-with-id.csv')


# In[136]:


#Getting song features
uris = list(songs_with_id.URI)
danceability_list = list()
energy_list = list()
key_list = list()
loudness_list = list()
mode_list = list()
speechiness_list = list()
acousticness_list = list()
instrumentalness_list = list()
liveness_list = list()
valence_list = list()
tempo_list = list()
duration_list = list()
time_signature_list= list()


# In[137]:


def get_audio_features(uri):
    search = sp.audio_features(uri)
    if search[0] == None:
        danceability_list.append(np.nan)
        energy_list.append(np.nan)
        key_list.append(np.nan)
        loudness_list.append(np.nan)
        mode_list.append(np.nan)
        speechiness_list.append(np.nan)
        acousticness_list.append(np.nan)
        instrumentalness_list.append(np.nan)
        liveness_list.append(np.nan)
        valence_list.append(np.nan)
        tempo_list.append(np.nan)
        duration_list.append(np.nan)
        time_signature_list.append(np.nan) 
        return ('Error on: ' + str(uri))
    
    search_list = search[0]
    
    danceability_list.append(search_list['danceability'])
    energy_list.append(search_list['energy'])
    key_list.append(search_list['key'])
    loudness_list.append(search_list['loudness'])
    mode_list.append(search_list['mode'])
    speechiness_list.append(search_list['speechiness'])
    acousticness_list.append(search_list['acousticness'])
    instrumentalness_list.append(search_list['instrumentalness'])
    liveness_list.append(search_list['liveness'])
    valence_list.append(search_list['valence'])
    tempo_list.append(search_list['tempo'])
    duration_list.append(search_list['duration_ms'])
    time_signature_list.append(search_list['time_signature'])


# In[139]:


token = util.oauth2.SpotifyClientCredentials(client_id='b7fc404a069d4a0f83049834a7d09a25', client_secret='990a5a3792e9469fac00c288ec610d7d')
cache_token = token.get_access_token()
spotify = spotipy.Spotify(cache_token)
sp = spotipy.Spotify(auth=cache_token)


# In[140]:


for i in tqdm_notebook(range(len(uris))):
    get_audio_features(uris[i])


# In[141]:


songs_with_id['Danceability'] = danceability_list
songs_with_id['Energy'] = energy_list
songs_with_id['Key'] = key_list
songs_with_id['Loudness'] = loudness_list
songs_with_id['Mode'] = mode_list
songs_with_id['Speechiness'] = speechiness_list
songs_with_id['Acousticness'] = acousticness_list
songs_with_id['Instrumentalness'] = instrumentalness_list
songs_with_id['Liveness'] = liveness_list
songs_with_id['Valence'] = valence_list
songs_with_id['Tempo'] = tempo_list
songs_with_id['Duration'] = duration_list
songs_with_id['Time_Signature'] = time_signature_list


# In[142]:


songs_with_id.dropna()


# In[143]:


songs_with_id.head()


# In[144]:


songs_with_id.tail()


# In[145]:


songs_with_id.to_csv('../data-files/songs-with-features.csv')


# In[ ]:




