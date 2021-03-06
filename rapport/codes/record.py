import pyaudio

FORMAT = pyaudio.paInt16 # 16 bits samples size
CHANNELS = 2 # Stereo
RATE = 44100 # Sampling rate
CHUNK = 1024
RECORD_SECONDS = 5 # Number of seconds to record

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("finished recording")
