# Testing creation of mrosterMonsterWidgetel filter bank

import numpy as np
import math

import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft, dct

import pylab as pl

def freqToMel(freq):
    return 1127 * math.log(1 + freq/700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel/1127.)-1.)

mel_filter = []
max_Hz = 22050
min_Hz = 0
num_filters = 26

min_mel = int(freqToMel(min_Hz))
max_mel = int(freqToMel(max_Hz/2))
mel_range = np.array(xrange(num_filters+2))

# Find the mel centers
mel_start = np.array(mel_range * (max_mel - min_mel)/ num_filters + min_mel)
mel_end = mel_start[1:]
mel_end = np.append(mel_end, max_mel)
mel_centers = np.array([(a+b)/2 for (a,b) in zip(mel_start, mel_end)])

# Find the corresponding frequencies
freq_start = np.floor(map(melToFreq, mel_start))
freq_end = np.ceil(map(melToFreq, mel_end))
freq_centers = np.round(map(melToFreq, mel_centers))

# Convert to int
freq_start = freq_start.astype(int)
freq_end = freq_end.astype(int)
freq_centers = freq_centers.astype(int)

# Make an interleaved array
freq_stacked = np.vstack((freq_centers, freq_end))
freq_stacked = freq_stacked.flatten('F')
freq_stacked = np.insert(freq_stacked, 0, 0)

num_points = 50

for i in xrange(num_filters):
    [start, center, end] = freq_stacked[i:i+3]

    x_range_1 = np.linspace(start, center, num=num_points)
    x_range_2 = np.linspace(center, end, num=num_points)
    x_range = np.append(x_range_1, x_range_2)

    y_range_1 = np.linspace(0.0, 1.0, num=num_points)
    y_range_2 = np.linspace(1.0, 0.0, num=num_points)
    y_range = np.append(y_range_1, y_range_2)

    mel_filter.append((x_range, y_range))
# Plot graphs 
pl.figure(1)
for (x_range, y_range) in mel_filter:
    pl.plot(list(x_range), list(y_range))

pl.suptitle("26 Triangular MFCC filters, 22050 Hz signal, Windowsize 1024")
pl.xlabel("Frequency")
pl.ylabel("Amplitude")
pl.xlim([0,300])
pl.savefig("plot2.png")

pl.show()

