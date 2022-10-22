import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

songDF = pickle.load(open('songDF.pkl','rb'))
allsongs = pickle.load(open('Data_all_songs.pkl','rb'))
complete_feature_set = pickle.load(open('complete_feature_set.pkl','rb'))
playlistDF_test = pickle.load(open('test_playlist.pkl','rb'))
playlistDF_test= playlistDF_test[playlistDF_test['name']=="Liked Songs"]
cosine = pickle.load(open('cosine (1).pkl','rb'))
playlist1 = pickle.load(open('playlistDF_test1.pkl','rb'))
playlist2 = pickle.load(open('playlistDF_test2.pkl','rb'))


def recommend_song(song, df_cosine_similarity, df, num_tracks):
  df.reset_index(drop=True,inplace =True)
  ind = df.index[df['track_name'] == song][0]
  uri = df['track_uri'].iloc[ind]
  index = df.index[df['track_uri'] == uri][0]
  similarities = df_cosine_similarity.iloc[:, index].sort_values(ascending=False)
  final_indices = list(similarities[1:num_tracks].index)
  songs_recommended = df[['track_name', 'artist_name','album_name']].iloc[final_indices]
  return songs_recommended


def generate_playlist_feature(complete_feature_set, playlist_df):
  complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
  complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
  complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "id")
  return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist




def generate_playlist_recos(df, features, nonplaylist_features):
  non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
  non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
  non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
  return non_playlist_df_top_40



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(
             "https://images.pexels.com/photos/92083/pexels-photo-92083.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


st.title('Music Recommendation System')


Choice = ['Song', 'Playlist']
selected_type = st.selectbox('SELECT ', Choice)

if selected_type == 'Song':
  selected_song = st.selectbox('SELECT A SONG', allsongs['track_name'].values)
  number = range(1, 20)
  select_number = st.selectbox('NUMBER OF SONGS', number)
  if st.button("RECOMMEND"):
    songs = recommend_song(selected_song, cosine, allsongs, select_number)
    st.write(songs.reset_index(drop=True))


else:
  values = ['Playlist1', 'Playlist2']
  selected_playlist = st.selectbox('SELECT PLAYLIST NAME', values)
  number = range(1, 20)
  select_number = st.selectbox('NUMBER OF SONGS', number)
  if selected_playlist =='Playlist1':
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(
      complete_feature_set, playlist1)
    recommend = generate_playlist_recos(songDF, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)
    recommended = recommend.head(select_number)
    listed_songs = recommended[['artist_name', 'track_name']]
    if st.button("RECOMMEND"):
      st.write(listed_songs.reset_index(drop=True))

  elif selected_playlist =='Playlist2':
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(
      complete_feature_set, playlist2)
    recommend = generate_playlist_recos(songDF, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)
    recommended = recommend.head(select_number)
    listed_songs = recommended[['artist_name', 'track_name']]
    if st.button("RECOMMEND"):
      st.write(listed_songs.reset_index(drop=True))

