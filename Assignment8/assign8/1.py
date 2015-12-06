# CS4347 Assignment 8   
# Synthesis beyond Two Sine Waves

import numpy as np

import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft

import pylab as pl

freq = 1000.
length = 1.
amp = 0.5
fs = 44100

time = np.arange(0, length, 1./fs)

def gen_perfect_sawtooth():
    sawtooth = amp*scipy.signal.sawtooth(freq*2.0*np.pi*time)
    write_audio(sawtooth, 'perfect_sawtooth.wav')

    return sawtooth

def gen_band_limited():
    M = 22
    wave = np.zeros(len(time))

    for i in xrange(1,M+1):
        temp_wave = (1./i)*np.sin(i*2.0*np.pi*time*freq)
        wave = wave + temp_wave
    
    wave = wave*(-2.0*amp/np.pi)

    write_audio(wave, 'constructed.wav')
    
    return wave

def write_audio(notes, filename):
    scaled = (notes*32767).astype(np.int16)
    scipy.io.wavfile.write(filename, fs, scaled)

def plot_graphs(perfect_sawtooth, gen_sawtooth):
    pl.figure()
    pl.plot(perfect_sawtooth, label="Perfect Sawtooth")
    pl.plot(gen_sawtooth, label="Reconstructed Sawtooth")
    pl.legend()

    pl.suptitle('Sawtooth wave reconstruction with 22 sine waves')
    pl.ylabel('Amplitude')
    pl.xlabel('Time')
    pl.xlim([0, 140])
    pl.ylim([-0.8,0.8])

    pl.savefig('sawtooth.png')
    pl.show()

def plot_db_graphs(perfect_power, gen_power):
    pl.figure()
    pl.plot(perfect_power, label="Perfect Sawtooth")
    pl.plot(gen_power, label="Reconstructed Sawtooth")
    pl.title('Sawtooth wave reconstruction with 22 sine wave')
    pl.xlabel('FFT bin')
    pl.ylabel('db')
    pl.xlim([0,4000])
    pl.legend()
    pl.savefig('db-mag-fft.png')
    pl.show()

def power_spectrun(x):
    frame_length = 8192
    x = x[:frame_length]
    window = np.blackman(len(x))
    fft = np.fft.fft(x*window)
    fft = fft[:len(fft)/2+1]
    magfft = np.abs(fft)/ (np.sum(window)/2.0)
    epsilon = 1e-10
    db = 20. * np.log10(magfft+epsilon)
    return db

if __name__ == "__main__":
    perfect_sawtooth = gen_perfect_sawtooth()
    gen_sawtooth = gen_band_limited()
    plot_graphs(perfect_sawtooth, gen_sawtooth)

    perfect_power = power_spectrun(perfect_sawtooth)
    gen_power = power_spectrun(gen_sawtooth)
    plot_db_graphs(perfect_power, gen_power)
