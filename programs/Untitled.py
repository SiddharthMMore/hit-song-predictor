#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np


# In[10]:


test1= pd.DataFrame(columns=['Title', 'Artist', 'Year'])


# In[11]:


years = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
url_template = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'


# In[12]:


for year in years:
    url = url_template + str(year)
    temp_df = pd.read_html(url, header=0)[0]
    temp_df.columns.values[2] = "Artist"
    temp_df = temp_df.drop(temp_df.columns[0], axis=1)
    temp_df['Artist']=temp_df['Artist'].astype(str)
    temp_df.Artist = [artist.split(' featuring')[0] for artist in temp_df.Artist]
    temp_df.Title = [title.strip('\"') for title in temp_df.Title]
    temp_df['Year'] = year
    test1 = test1.append(temp_df)


# In[13]:


test1.head()


# In[14]:


test1.tail()


# In[15]:


test1.shape


# In[18]:


test1 = test1.reset_index(drop=True)
test1['Year'] = pd.to_numeric(test1['Year'])
test1['Title'] = test1['Title'].astype(str)
test1['Artist'] = test1['Artist'].astype(str)


# In[19]:


test1=test1.drop('Year',axis=1)
test1 = test1.drop_duplicates(subset="Title", keep="first")


# In[20]:


import spotipy
import spotipy.util as util


# In[21]:


def get_spotify_uri(title, artist):
    query = title + " " + artist
    search = sp.search(q=query, limit=50, offset=0, type='track', market='US')
    search_items = search['tracks']['items']
    for i in range(len(search_items)):
            uri = search_items[i]['id']
            return uri
    return 0


# In[22]:


token = util.oauth2.SpotifyClientCredentials(client_id='b7fc404a069d4a0f83049834a7d09a25', client_secret='990a5a3792e9469fac00c288ec610d7d')
cache_token = token.get_access_token()
spotify = spotipy.Spotify(cache_token)
sp = spotipy.Spotify(auth=cache_token)


# In[25]:


test1['Top100']=1
titles = list(test1.Title)
artists = list(test1.Artist)
top100 = list(test1.Top100)
spotify_uri = list()
errors = list()


# In[26]:


temp1 = list()


# In[30]:


from tqdm import tqdm_notebook
for i in tqdm_notebook(range(0,1000)):
    uri = get_spotify_uri(titles[i], artists[i])
    
    if uri != 0:
        temp1.append(uri)
    else:
        temp1.append(uri)
        errors.append(i)
spotify_uri = spotify_uri + temp1


# In[32]:


temp1


# In[34]:


len(temp1)


# In[35]:


spotify_uri = spotify_uri + temp1


# In[37]:


songs_with_id = pd.DataFrame()
songs_with_id['Title'] = titles;
songs_with_id['Artist'] = artists;
songs_with_id['Top100'] = top100;
songs_with_id['URI'] = spotify_uri;
songs_with_id=songs_with_id[songs_with_id.URI!=0]


# In[38]:


songs_with_id.shape


# In[39]:


songs_with_id.reset_index(drop=True,inplace=True)


# In[40]:


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


# In[41]:


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


# In[42]:


token = util.oauth2.SpotifyClientCredentials(client_id='b7fc404a069d4a0f83049834a7d09a25', client_secret='990a5a3792e9469fac00c288ec610d7d')
cache_token = token.get_access_token()
spotify = spotipy.Spotify(cache_token)
sp = spotipy.Spotify(auth=cache_token)


# In[43]:


for i in tqdm_notebook(range(len(uris))):
    get_audio_features(uris[i])


# In[44]:


songs_with_id.dropna()


# In[45]:


df_main = pd.read_csv('../data-files/songs-with-features.csv',index_col=0)


# In[52]:


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


# In[53]:


songs_with_id.dropna()


# In[62]:


combined_df = pd.concat([df_main, songs_with_id], ignore_index=True)


# In[63]:


combined_df = combined_df.drop_duplicates(subset="Title", keep="last")


# In[64]:


combined_df.tail()


# In[57]:


songs_with_id.head()


# In[58]:


df_main.head()


# In[66]:


combined_df.tail()


# In[67]:


combined_df.to_csv('../data-files/appended_songs.csv')


# In[ ]:




