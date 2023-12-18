import os
from pytube import YouTube


DIR = os.path.dirname(os.path.realpath(__file__))
path = lambda name: os.path.join(DIR, name)

def download_video_from_youtube(link, path):
    yt = YouTube(link)
    video = yt.streams.get_highest_resolution()
    # download the video
    video.download(path)
# example usage:
download_video_from_youtube('https://www.youtube.com/watch?v=1234567', path('./'))
