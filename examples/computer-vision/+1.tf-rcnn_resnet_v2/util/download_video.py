from pytube import YouTube

def Download(link, output_path):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(output_path)
    except:
        print("An error has occurred")
    print("Download is completed successfully")


link = input("Enter the YouTube video URL: ")
Download(link, '../data/tmp')
