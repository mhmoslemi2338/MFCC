import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

########################################
############### part 1 #################
########################################
N = 31
n = np.arange(N)

W_R = np.ones([N])

W_T = 2 * n / (N - 1)
W_T[(N - 1) // 2:] = 2 - W_T[(N - 1) // 2:]

W_H = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

W_HN = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

W_B = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))


###### save result ######
fig = plt.figure(figsize=[12, 6])
plt.plot(n, W_R, color='r', label='Rectangular')
plt.plot(n, W_T, color='g', label='Triangular')
plt.plot(n, W_H, color='b', label='Hamming')
plt.plot(n, W_HN, color='m', label='Hanning')
plt.plot(n, W_B, color='k', label='Blackman')
plt.xlabel('n')
plt.grid()
plt.legend()
fig.savefig('Question1_a.jpg', dpi=2 * fig.dpi)
plt.close(fig)


########################################
############### part 2 #################
########################################
FFT_N = 1024
Fs = 10000

x = np.linspace(0.0, FFT_N / Fs, FFT_N, endpoint=False)
freq = fftshift(fftfreq(FFT_N, 1 / Fs))

W_R_fft = fftshift(fft(W_R, FFT_N))
W_T_fft = fftshift(fft(W_T, FFT_N))
W_H_fft = fftshift(fft(W_H, FFT_N))
W_HN_fft = fftshift(fft(W_HN, FFT_N))
W_B_fft = fftshift(fft(W_B, FFT_N))


###### save result ######
fig = plt.figure(figsize=[15, 10])
plt.plot(freq / Fs, np.log10(np.abs(W_R_fft)), color='r', label='Rectangular')
plt.plot(freq / Fs, np.log10(np.abs(W_T_fft)), color='g', label='Triangular')
plt.plot(freq / Fs, np.log10(np.abs(W_H_fft)), color='b', label='Hamming')
plt.plot(freq /Fs,np.log10(np.abs(W_HN_fft) +0.000001),color='m',label='Hanning')
plt.plot(freq /Fs,np.log10(np.abs(W_B_fft) +0.000001),color='k',label='Blackman')

plt.title(' 1024 point FFT for differetn windows ', fontsize=16)
plt.ylabel(r'$log_{10}\,|H ( e^{j\omega})| $', fontsize=20)
plt.xlabel(r'$\frac{frequency}{F_s}$', fontsize=24)
plt.grid()
plt.legend()
fig.savefig('Question1_b.jpg', dpi=2 * fig.dpi)
plt.close(fig)
