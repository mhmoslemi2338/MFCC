
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, fftshift
from scipy.fftpack import idct
import seaborn as sns

def make_filter_bank(Fs, freq_l=50, freq_h=4000, N_filter=40):
    freq_l_mel = (2595 * np.log10(1 + freq_l / 700))         # Convert Hz to Mel
    freq_h_mel = (2595 * np.log10(1 + freq_h / 700))         # Convert Hz to Mel
    mel = np.linspace(freq_l_mel, freq_h_mel, N_filter +2)        # Equally spaced in Mel scale
    freq = 700 * (np.power(10, mel / 2595) - 1)                 # Convert Mel to Hz
 
    filter_bank = np.zeros([N_filter, Fs])
    bins = np.round(freq / freq_h * Fs)
    for m in range(1, N_filter + 1):
        for i in range(Fs):
            p1 = bins[m - 1]
            p2 = bins[m]
            p3 = bins[m + 1]
            if p1 <= i <= p2:
                filter_bank[m - 1][i] = (i - p1) / (p2 - p1)
            elif p2 <= i <= p3:
                filter_bank[m - 1][i] = (p3 - i) / (p3 - p2)
    return filter_bank


def make_MFCC(name):
    MFCC_all = []
    for index in range(1, 7):

        Fs, data = wavfile.read('data/%s_P0%d.wav' %(name, index))

        frame_length = int(round(0.03 * Fs))
        frame_step = int(round(frame_length / 2))
        num_frames = np.ceil(len(data) / frame_step)
        pad = num_frames * frame_step - len(data)
        data = np.append(data, np.zeros(int(pad)))

        W_H = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
        filter_bank = make_filter_bank(Fs, freq_l=50, freq_h=4000, N_filter=40)

        Frames = []
        Frames_fft = []
        for i in range(0, len(data) - frame_length, frame_step):
            window = data[i:i + frame_length]
            Frames.append(np.multiply(W_H, window))
            Frames_fft.append(fftshift(fft(Frames[-1], Fs)))
        Frames = np.array(Frames)
        Frames_fft = np.array(Frames_fft)
        Frames_power = np.power(np.abs(Frames_fft), 2)

        Frames_Filtered = np.dot(Frames_power, (filter_bank.T))
        Frames_Filtered = np.log10(Frames_Filtered)

        mfcc0 = idct(Frames_Filtered)[:, 1:13]
        delta = np.diff(mfcc0, axis=0, n=1)
        delta_delta = np.diff(mfcc0, axis=0, n=2)

        MFCC = []
        for i, row in enumerate(mfcc0[2:, :]):
            MFCC.append(np.concatenate([row, delta[i + 1], delta_delta[i]]))
        MFCC = np.array(MFCC)
        MFCC_all.append(MFCC)

    MFCC = MFCC_all[0]
    for i in range(1, len(MFCC_all)):
        MFCC = np.concatenate([MFCC, MFCC_all[i]])
    return MFCC






MFCC1=make_MFCC('S01')
MFCC2=make_MFCC('S02')



########### save result #########
xticks = np.int32(np.linspace(0, len(MFCC1) , 12)).tolist()
xticklabels = np.linspace(0, len(MFCC1), 12) * 0.015
xticklabels = np.round(xticklabels , 3).tolist()
yticks = np.linspace(1, 36, 12, dtype=np.int)

fig = plt.figure(figsize=(18, 8))
sns.heatmap(np.flip(MFCC1).T, cmap='coolwarm')#, cbar=False)
plt.xticks(xticks, xticklabels, rotation=0, fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.yticks(yticks, np.flip(yticks), rotation=0, fontsize=15)
plt.ylabel('MFCC Coeffs', fontsize=20)
plt.tight_layout()
fig.savefig('MFCC_speaker1.jpg', dpi=7 * fig.dpi)
plt.close(fig)



########### save result #########
xticks = np.int32(np.linspace(0, len(MFCC2) , 12)).tolist()
xticklabels = np.linspace(0, len(MFCC2), 12) * 0.015
xticklabels = np.round(xticklabels , 3).tolist()
yticks = np.linspace(1, 36, 12, dtype=np.int)

fig = plt.figure(figsize=(18, 8))
sns.heatmap(np.flip(MFCC2).T, cmap='coolwarm')#, cbar=False)
plt.xticks(xticks, xticklabels, rotation=0, fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.yticks(yticks, np.flip(yticks), rotation=0, fontsize=15)
plt.ylabel('MFCC Coeffs', fontsize=20)
plt.tight_layout()
fig.savefig('MFCC_speaker2.jpg', dpi=7 * fig.dpi)
plt.close(fig)
