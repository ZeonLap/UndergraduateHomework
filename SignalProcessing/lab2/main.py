import librosa
import numpy as np
import argparse
from matplotlib import pyplot as plt

def get_key_tone(sp, interval):
    _map = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#']
    # print(np.sum(np.abs(sp) ** 2))
    if np.sum(np.abs(sp) ** 2) < 20000:
        return '-1'
    
    # x = np.linspace(0, 375 * interval, 376)
    # plt.plot(x, np.abs(sp))
    # plt.xlim(0, 2000)
    # plt.show()
    
    sorted_freq = np.argsort(np.abs(sp))[::-1]

    max_freq = np.sort(sorted_freq[0:2] * interval)

    distance = np.array([0] * 12)
    distance[0] = np.abs(max_freq[0] - 697) + np.abs(max_freq[1] - 1209)
    distance[1] = np.abs(max_freq[0] - 697) + np.abs(max_freq[1] - 1336)
    distance[2] = np.abs(max_freq[0] - 697) + np.abs(max_freq[1] - 1477)
    distance[3] = np.abs(max_freq[0] - 770) + np.abs(max_freq[1] - 1209)
    distance[4] = np.abs(max_freq[0] - 770) + np.abs(max_freq[1] - 1336)
    distance[5] = np.abs(max_freq[0] - 770) + np.abs(max_freq[1] - 1477)
    distance[6] = np.abs(max_freq[0] - 852) + np.abs(max_freq[1] - 1209)
    distance[7] = np.abs(max_freq[0] - 852) + np.abs(max_freq[1] - 1336)
    distance[8] = np.abs(max_freq[0] - 852) + np.abs(max_freq[1] - 1477)
    distance[9] = np.abs(max_freq[0] - 941) + np.abs(max_freq[1] - 1209)
    distance[10] = np.abs(max_freq[0] - 941) + np.abs(max_freq[1] - 1336)
    distance[11] = np.abs(max_freq[0] - 941) + np.abs(max_freq[1] - 1477)

    return _map[np.argmin(distance)]
    

def key_tone_recognition(audio_array):
    '''
        请大家实现这一部分代码
    '''
    audio_array = audio_array[0]
    f = 48000
    N = 750
    interval = f / N
    nframes = int(audio_array.shape[0] / 750)
    for i in range(nframes):
        frame = audio_array[i * 750 : (i + 1) * 750]
        sp = np.fft.rfft(frame)
        print(get_key_tone(sp, interval), end = ' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type = str, help = 'test file name', required = True)
    args = parser.parse_args()
    input_audio_array = librosa.load(args.audio_file, sr = 48000, dtype = np.float32) # audio file is numpy float array
    key_tone_recognition(input_audio_array)