import os
import yt_dlp

def download_specific_songs(url_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
        
    # Configuration: High Quality WAV
    download_opts = {
        'format': 'bestaudio/best',
        # Name files by their YouTube Title to keep them organized
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # "ignoreerrors": True -> skips a video if it's region-locked/deleted 
        # instead of crashing the whole script
        'ignoreerrors': True, 
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download(url_list)
            
    print(f"\nAll available links saved to: {output_dir}")


if __name__ == "__main__":
    # NOTE: only need large dataset for full dream songs. - aim for maybe 2-3 hours of data
    # only need the trap part to train the section kmeans classifier for trap! - aim for maybe 150 segments total
    
    os.chdir(r"C:\Users\adamy\PycharmProjects\ML 1\dream_voice\songScrape")
    SAVE_DIR_DREAM = "./rawData/fullDreamSongs"
    SAVE_DIR_TRAP = "./rawData/fullTrapSongs"
    
    print(os.getcwd())
    # Read from file
    with open("./trap_songs.txt", "r") as f:
        # .strip() removes newlines/spaces
        MY_DREAMY_SONGS = [line.strip() for line in f if line.strip()]
    
    # download_specific_songs(MY_DREAMY_SONGS, SAVE_DIR_DREAM)
    download_specific_songs(MY_DREAMY_SONGS, SAVE_DIR_TRAP)

    os.chdir(r"C:\Users\adamy\PycharmProjects\ML 1")