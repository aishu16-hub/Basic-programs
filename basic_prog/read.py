#To read a csv file
import pandas as pd

df = pd.read_csv('sample.csv')
print(df)



#To read a jpg file
from PIL import Image

# Open image file
img = Image.open('images.jpg')  

# Show image
img.show()

# Access image properties
print(img.format)
print(img.size)
print(img.mode)



#To read png file
from PIL import Image

# Open image file
img = Image.open('images.png')  

# Show image
img.show()

# Access image properties
print(img.format)
print(img.size)
print(img.mode)




#To read wav file
import wave

# Open the WAV file
with wave.open('pure-tone.wav', 'rb') as wav_file:
    # Get number of channels
    channels = wav_file.getnchannels()
    # Get sample width
    sampwidth = wav_file.getsampwidth()
    # Get frame rate (sampling frequency)
    framerate = wav_file.getframerate()
    # Get total number of frames
    nframes = wav_file.getnframes()

    # Read all frames
    frames = wav_file.readframes(nframes)

    # Print some info
    print(f"Channels: {channels}")
    print(f"Sample width: {sampwidth}")
    print(f"Frame rate: {framerate}")
    print(f"Total frames: {nframes}")
