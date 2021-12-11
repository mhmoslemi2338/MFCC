

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import idct

# This file is exaclty the same as the q3.py which we 
# implemented the MFCC extractor so we do not put comment
# for it.
#
# There is a tiny difference in the make_MFCC function which is
# now we have an opion to concatenate the delta and delta_delta values
# to the MFCC's , which declares with an boolean flag



def make_filter_bank(length, freq_l=50, freq_h=4000, N_filter=40):
    freq_l_mel = (2595 * np.log10(1 + freq_l / 700))         # Convert Hz to Mel
    freq_h_mel = (2595 * np.log10(1 + freq_h / 700))         # Convert Hz to Mel
    # Equally spaced in Mel scale# Convert Mel to Hz
    mel = np.linspace(freq_l_mel, freq_h_mel, N_filter + 2)
    freq = 700 * (np.power(10, mel / 2595) - 1)
    filter_bank = np.zeros([N_filter, length])
    bins = np.round(freq / freq_h * length)
    for m in range(1, N_filter + 1):
        for i in range(length):
            p1 = bins[m - 1]
            p2 = bins[m]
            p3 = bins[m + 1]
            if p1 <= i <= p2:
                filter_bank[m - 1][i] = (i - p1) / (p2 - p1)
            elif p2 <= i <= p3:
                filter_bank[m - 1][i] = (p3 - i) / (p3 - p2)
    return filter_bank


def make_MFCC(name, append_delta=True):
    MFCC_all = []
    for index in range(1, 7):
        Fs, data = wavfile.read('data/%s_P0%d.wav' % (name, index))
        frame_length = int(round(0.03 * Fs))
        frame_step = int(round(frame_length / 2))
        num_frames = np.ceil(len(data) / frame_step)
        pad = num_frames * frame_step - len(data)
        data = np.append(data, np.zeros(int(pad)))
        W_H = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
        filter_bank = make_filter_bank(Fs, freq_l=0, freq_h=Fs / 2, N_filter=40)
        Frames = []
        Frames_fft = []
        for i in range(0, len(data) - frame_length, frame_step):
            window = data[i:i + frame_length]
            Frames.append(np.multiply(W_H, window))
            Frames_fft.append(np.fft.fft(np.multiply(W_H, window), Fs))
        Frames = np.array(Frames)
        Frames_fft = np.array(Frames_fft)
        Frames_power = np.power(np.abs(Frames_fft), 2)
        Frames_Filtered = np.dot(Frames_power, (filter_bank.T))
        Frames_Filtered = np.log(Frames_Filtered)

        mfcc0 = idct(Frames_Filtered, type=2, axis=1, norm='ortho')[:, 0:10]

        delta = np.zeros(mfcc0.shape)
        padded = np.pad(mfcc0, ((1, 0), (0, 0)), mode='edge')
        for t in range(len(mfcc0)):
            delta[t] = np.dot(np.array([-1, 1]), padded[t: t + 2])

        delta_delta = np.zeros(delta.shape)
        padded = np.pad(delta, ((1, 0), (0, 0)), mode='edge')
        for t in range(len(delta)):
            delta_delta[t] = np.dot(np.array([-1, 1]), padded[t: t + 2])

        if append_delta:
            mfcc0 = (np.concatenate([mfcc0.T, delta.T, delta_delta.T])).T
        MFCC_all.append(mfcc0)
    return MFCC_all
