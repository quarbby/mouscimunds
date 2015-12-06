# CS4347 Assignment 9   
# Analysis of Speech

import numpy as np

import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft

import pylab as pl

fs = 22050

def analyze_audio(audiofile):
    num_frames = 463
    window_size = 128

    wav = scipy.io.wavfile.read(audiofile)
    wav_float = wav[1]/ 32768.0

    spec = []
    freq_amp = []

    for i in xrange(num_frames):
        start = i * window_size
        end = start + window_size
        buff_data = wav_float[start:end]

        if not len(buff_data) < window_size:
            hamming = apply_hamming_window(buff_data)
            fft_signal = perform_fft(hamming)

            spec.append(fft_signal)

            max_val = np.max(fft_signal)
            max_val_index = np.argmax(fft_signal).astype(float)
            freq = max_val_index / window_size *fs
           
            freq_amp.append((freq, max_val))

    freq_amp = np.array(freq_amp)
    np.savetxt('freq_amp.csv', freq_amp, fmt='%.6g', delimiter=',')
            
    return spec, freq_amp

def synthesize(freq_amp):
    window_size = 128
    fs = 22050

    reconstructed = []
    re_spec = []

    for (freq, amp) in freq_amp:
        sine_wave = amp * np.sin(2 * np.pi * freq/ fs * np.arange(window_size))
        reconstructed = np.concatenate([reconstructed, sine_wave])

        hamming = apply_hamming_window(sine_wave)
        fft_signal = perform_fft(hamming)
        re_spec.append(fft_signal)

    re_wav_norm = reconstructed / reconstructed.max() * 32767
    re_wav_norm = re_wav_norm.astype(np.int16)
    scipy.io.wavfile.write('reconstructed.wav', fs, re_wav_norm)

    return re_spec

def plot_spectrograms(spec, re_spec):
    spec = np.array(spec)
    re_spec = np.array(re_spec)

    pl.figure()
    pl.suptitle("Spectrogram", fontsize=20)

    ax1 = pl.subplot(2,1,1)
    pl.imshow(spec.T/spec.max(), 
            origin='lower', aspect='auto')
    ax1.set_ylabel('Freq bin')
    ax1.set_xlabel('Frames (Original wav)')

    ax2 = pl.subplot(2,1,2)
    pl.imshow(re_spec.T/re_spec.max(),
            origin='lower', aspect='auto')
    ax2.set_ylabel('Freq bin')
    ax2.set_xlabel('Frames (Reconstructed wav)')

    pl.savefig('spectrogram.png')
    #pl.show()

def apply_hamming_window(buff_data):
    window = scipy.signal.hamming(len(buff_data))
    return [a*b for a,b in zip(list(window), list(buff_data))]

def perform_fft(buff_data):
    signal = scipy.fftpack.fft(buff_data)
    return np.abs(signal[:len(buff_data)/2+1])

if __name__ == "__main__":
    spec, freq_amp = analyze_audio("./clear_d1.wav")
    re_spec = synthesize(freq_amp)
    plot_spectrograms(spec, re_spec)
