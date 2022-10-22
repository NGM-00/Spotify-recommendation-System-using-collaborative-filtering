# SPOTIFY RECOMMENDATION SYSTEM<br>

<img height="200px" src="https://cdn.wccftech.com/wp-content/uploads/2021/07/Spotify.jpg">
 
### PLANNING STAGE OF SYSTEM<br>
In This Project, we build a Spotify Recommendation System as a Machine Learning application. Spotify is a digital music, podcast, and video service that gives us access to millions of songs using Spotify API.
### WHAT IS A RECOMMENDATION SYSTEM ?
- A Recommendation System is a filtering system which aim is to predict a rating or preference a user would give to an item, eg. a film, a product, a song, etc.
- Spotify use different types of Recommendation Systems which are:
    - Collaborative Filtering Algorithm
    - Content Based Filtering
    - Natural Language Processing
- In this project we build our Recommendation System to recommend similar songs to the user input, based Collaborative Filtering with clustering.

### DATA EXTRACTION<br>
Data came from 2 sources:
- API Calls of Spotify's Web API to get audio features for each track.
- The Spotify Million Playlist Dataset which contained Four separate JSON files.

### UNDERSTANDING THE DATA:
- The Spotify million playlist dataset consists of 4 JSON Files:
    - pid - the playlist ID.
    - name - the name of the playlist
    - tracks - an array of tracks that are in the playlist. Each element of this array contains the following fields:
         * pos - the position of the track in the playlist (zero offset)
         * track_name - the name of the track
         * track_uri - the Spotify URI of the track
         * artist_name - the name of the primary artist of the track
         * artist_uri - the Spotify URI of the primary artist of the track
         * album_name - the name of the album that the track is on
         * album_uri -- the Spotify URI of the album that the track is on
         * duration_ms - the duration of the track in milliseconds
    - num_tracks - the total number of tracks in the playlist.
- The Audio Features are scraped from Spotify API using spotipy library, which are:
    - Acousticness- A confidence measure from 0.0 to 1.0 of whether the track is acoustic or not.
    - Danceability- A value of 0.0 is least danceable and 1.0 is most danceable.
    - Duration_ms- The duration of the track in milliseconds.
    - Energy- It is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. 
    - Id-  Spotify ID for the track.
    - Instrumentalness- Measure of vocals in the track.
    - Key- Integers map to pitches using standard Pitch Class notation. If no key was detected, the value is -1.
    - Liveness- Detects the presence of an audience in the recording. A value above 0.8 provides strong likelihood that the track is live.
    - Loudness- It is the quality of a sound, correlate of amplitude and value ranges between -60 and 0 db.
    - Mode- Indicates the modality of a track. Major is represented by 1 and minor is 0.
    - Speechiness- It detects the presence of spoken words in a track. Values between 0 and 1.
    - Tempo- It is the speed or pace of a given piece and derives directly from the average beat duration.
    - Time_signature- It is the measure of each beat in a bar. The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".
    - Valence- It is a measure of positivity with high valence sound more positive (e.g. happy, cheerful),low valence sound more negative (e.g. sad,angry).

## DATA CLEANING <br>
- Extracted data had variables like "Description" which had more than 40% null values. Hence, they were dropped. 
- Features with less than 40% null values were filled with median values in continous value type features where as that of mode values in object-type features.
- Duplicates rows were identified and dropped from the dataset.

## EXPLORATORY DATA ANALYSIS <br>

<img height="300px" src="https://github.com/Samrat-Doi/Spotify-Million-Songs-Playlist-Collaborative-Filtering-Method/blob/main/int_visualization_1.png">
 
   Integer columns Skewness: 
    --------------------------------------------------------------------
    
            - 'pid': 0.017808960867489464, 
            - 'modified_at': -1.5708263246583083,
            - 'num_tracks': 0.4061981401693917,
            - 'num_albums': 0.6532191347624093,
            - 'num_followers': 57.32408837141191,
            - 'num_edits': 1.7343797592157533,
            - 'playlist_duration_ms': 0.5236201051400035,
            - 'num_artists': 0.9187385911738083,
            - 'pos': 1.195362099841709,
            - 'track_duration_ms': 13.061421293930474
    --------------------------------------------------------------------
-As per the above graphs:
   - "modified_at" feature seems to be left skewed.
   - "num_of_edits","liveness" and "speechiness" feature are right skewed.
 *Hence we can conclude, these feature need to be normalized.*
 
 <img height="300px" src="https://github.com/Samrat-Doi/Spotify-Million-Songs-Playlist-Collaborative-Filtering-Method/blob/main/comparision_visualization_3.png">
-In the above picture you can see the distribution of skewed features before and after applying the transformation methods.
   - power of 1/4 was applied to right skewed features.
   - cube root was applied to left skewed features.
   - 
 <img height="500px" src="https://github.com/Samrat-Doi/Spotify-Million-Songs-Playlist-Collaborative-Filtering-Method/blob/main/heatmap.png">
         --------------------------------------------------------------------
       
            ('num_albums', 'num_tracks', 0.8415447166107518),
            ('num_edits', 'num_tracks', 0.5299566460316026),
            ('num_edits', 'num_albums', 0.6163409150659508),
            ('playlist_duration_ms', 'num_tracks', 0.977662549636656),
            ('playlist_duration_ms', 'num_albums', 0.8319682926543831),
            ('playlist_duration_ms', 'num_edits', 0.5237533705040985),
            ('num_artists', 'num_tracks', 0.7269838343923531),
            ('num_artists', 'num_albums', 0.9355542001527783),
            ('num_artists', 'num_edits', 0.5776211268126814),
            ('num_artists', 'playlist_duration_ms', 0.7152958361571978),
            ('pos', 'num_tracks', 0.6524721120620393),
            ('pos', 'num_albums', 0.5490635615366086),
            ('pos', 'playlist_duration_ms', 0.6378924301360763),
            ('loudness', 'energy', 0.7507065639450008)

        --------------------------------------------------------------------
- Strong Positive Co-Relations between Features with threshold =  0.5
- Strong Negative Co-Relations between Features with threshold =  -0.5
*Hence, 'acousticness' is negatively correlated with 'energy' and 'loudness'.*

## FEATURE ENGINEERING AND PREPROCESSING <br>
- "Polarity" feature was created using textblob to calculate the opinion of the tracks using track_name.
- "Subjectivity" feature was created using textblob to calculate the intensity of opinion of the tracks using track_name.

## DESIGN AND PROTOTYPE<br>
-The first approach of memory concentrates on the computation of relationships across products or items separately.
- We will concentrate on Collaborative Filtering, which can be globally thought as a matrix-completion problem.
- Requirements:
     - Functional requirements:
        * The input will be the song name and the artist's name. The recommendation model goal/output is to find the top ten related songs to the input. 
        * It analyzes the input and then uses the provided Spotify dataset/API to get the output.
     -Non-functional requirements:
        * The model should obtain the highest accuracy possible by getting the most related songs to input. 
        * The model should be optimized to require as low computational power as possible. The model performance will be then evaluated using evaluation metrics. 

### MODEL BUILDING<br>
  **Choosing the features**
    - Correlation is used on reducing the tracks dataset features to the most important one.
    - PCA is used on reducing the most important features into other pca components that maintains the highest variability of the data.
    
  **Clustering**
    - Top 4 PCA are used for clustering as they contain 97% of data variations.
    - Based on it, K-Nearest Neighbours is used to group the data into clusters.
    - These clusters are used to predict and group testing data and make recommendation based on it.
    
  **Similarity Function**
 - We used track features and created the track metadata feature which includes the the "artist", "album" and "track name" and got the mean of the similarity
 (using cosaine similarity) due to track features and track metadata feature and recommend to the user according to similarity and popularity.
 - It is used with KNN to increase the accuracy of model.
 
<img height="350px" src="https://www.tyrrell4innovation.ca/wp-content/uploads/2021/06/rsz_jenny_du_miword.png">

#### OUTPUT:
 - When model is given input of a playlist ID, it gives a dataframe of tracks based on clustering of tracks in playlist and its cosine similarity of audio features with the tracks present in the dataset.
 - It also generates playlist based on a single track selected by the user.

## TESTING STAGE<br>
**Problem:** Our dataset doesn't has ratings feature which it difficult to measure the user preference directly.
**Solution:** We evaluated the model based on other features present in the playlist and used evaluation metrics to calculate its accuracy.

### TYPES OF EVAULATION METRICS
-  There are multiple evaluation metrics that can be used to measure accuracy of recommender system.
-  So, We should maximize all of them and solve the trade-off that happens between them, because some metrics are oposite to others but both of them is desired.

- *Classification Accuracy Metrics*   
- *Personalization*

#### CLASSIFICATION ACCURACY METRICS:
- This type of metrics measures whether this recommender system can recommend correct tracks to correct user.
- The exact rating or ranking of objects is ignored.

**F1-score**: 
- It's a formula that compines the above two metrics [Precision, recall].
- According to the model, f1-score of model is 0.76.
   
<img height="200px" src="https://3.bp.blogspot.com/--jLXutUe5Ss/VvPIO6ZH2tI/AAAAAAAACkU/pvVL4L-a70gnFEURcfBbL_R-GnhBR6f1Q/s1600/ConfusionMatrix.png">
   
**Matthews correlation coefficient (MCC)**: 
- It overcomes the drawbacks of F1 score.
- As we see in our calculations that F1 score doesn't consider True negative items.
- True negative is a term called for those items which recommender system does't put into his list because these items are not similar to the user.

<img height="50px" src="https://miro.medium.com/max/1400/1*R6_BTaMSdCLdNBa0oauFQQ.png">

#### PERSONALIZATION
-This metric wants to ensure that these certain items are recommended to this specific user.
- First, apply recommender system on all your users.
- Then, concatenate all the recommended items to gether with no dublicates in one list.
- Create a DataFrame so that columns are the recommended items, index is user list.
- Calculate the disimilarity between users.
- Model gives recommendations closer to the songs and playlist selected.
    
## DEPLOYMENT<br>
- To deploy the recommendation system we have to build a web app on local host first then use a cloud platform to deploy it on web.
- Building web app on local host can be done using Flask, html, css, reactJS, Streamlit etc.  
  
   **FLASK**
   - It is a web framework that provides libraries to build lightweight web applications in python. 
   - It is based on WSGI toolkit and jinja2 template engine.
   -  Flask is considered as a micro framework.
   - We create our app by using flask.
   
   **STREAMLIT**
   - Streamlit is an open source app framework in Python language. 
   - It helps us create web apps for data science and machine learning in a short time. 
   - It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.


- To deploy the app on cloud platform following files are created and uploaded on github:
      - Procfile: contains run statements for app file and setup.sh.
      - Requirements.txt: contains the libraries must be downloaded by Heroku to run app file (app.py) successfully, which is create using following command: 
           *pip freeze > requirement.txt*
      - App.py: contains the python code of the recommendation system algorithm.
      - Data files: data files like playlist data and dataset to recommend songs from.
   
   **Heroku**
   - Heroku is a platform as a service (PaaS) that enables developers to build, run, and operate applications entirely in the cloud [Heroku.com.].
   - It is directly linked to github repository then the branch containing all reqiured files is deployed.
   - It builds the dependencies for the app and back-trace the files required and on successful build, app is created.

    **AWS Elastic Beanstalk**
    - It is a free service for a limited period of time. 
    - It requires requiremts.txt, procfile and a new folder in app’s directory with the name “.ebextensions” and a new file named as “python.config” in it.
    - Inside this file add option_settings:  "AWS:elasticbeanstalk:container:python", WSGIPath: application:application and save file.
    - Now add all the files (main python file, flask folders (templates,static..), app data(dataframe,database…), requirements.txt, .ebextensions folder) into single “.zip” archive file.
    - Now deploy the application on AWS Elastic Beanstalk.
 
<img height="300px" src="https://github.com/Samrat-Doi/Spotify-Million-Songs-Playlist-Collaborative-Filtering-Method/blob/main/Screenshot%20(14).png">

**APPLICATIONS**

<a href="https://music-recommendation-1.herokuapp.com/" target="_blank">Spotify-Recommendation_System1</a>
</tr>
<a href="http://spotify-env.eba-pm2exfpy.us-east-1.elasticbeanstalk.com/" target="_blank">Spotify-Recommendation_System2</a>
