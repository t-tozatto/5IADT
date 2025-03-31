import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import json

SPOTIFY_CLIENT_ID = "CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "CLIENT_SECRET"
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"

YOUTUBE_API_KEY = "API_KEY"
YOUTUBE_HISTORY_URL = "https://www.googleapis.com/youtube/v3/videos"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-read-recently-played"
))

def get_spotify_history():
    results = sp.current_user_recently_played(limit=50)
    tracks = []
    for item in results['items']:
        track = item['track']
        tracks.append({
            'title': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'played_at': item['played_at']
        })
    return tracks

def get_youtube_liked_videos():
    params = {
        "part": "snippet,contentDetails",
        "myRating": "like",
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(YOUTUBE_HISTORY_URL, params=params)
    data = response.json()
    videos = []
    for item in data.get("items", []):
        videos.append({
            'title': item['snippet']['title'],
            'channel': item['snippet']['channelTitle'],
            'videoId': item['id']
        })
    return videos

spotify_data = get_spotify_history()
youtube_data = get_youtube_liked_videos()

# Salvar os dados em JSON
with open("music_data.json", "w", encoding="utf-8") as f:
    json.dump({"spotify": spotify_data, "youtube": youtube_data}, f, indent=4, ensure_ascii=False)
