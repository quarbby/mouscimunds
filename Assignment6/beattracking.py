# Assignment 6: Beat tracking

import numpy as np
import math

import scipy
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal

import pylab as pl

def read_wav_file(filename):
    wav = scipy.io.wavfile.read(filename)
    return wav[1]/32768.0

def create_buffer_matrix(wav):
    buff_length = 1024
    num_buffers = int(len(wav)/buff_length)*2

    hopsize = buff_length/2
    accent_signal_list = []

    for i in xrange(num_buffers):
        start = i * hopsize
        end = start + buff_length
        buff_data = np.array(wav[start:end])

        if not len(buff_data) < buff_length:
            transformed_signal  = accent_signal(buff_data)
            accent_signal_list.append(transformed_signal)

    return calc_accent_val(accent_signal_list)

def accent_signal(buff_data):
    hamming_signal = apply_hamming_window(buff_data)
    fft_signal = perform_fft(hamming_signal)

    return fft_signal

def apply_hamming_window(buff_data):
    window = scipy.signal.hamming(len(buff_data))
    return [a*b for a,b in zip(list(window), list(buff_data))]

def calc_accent_val(accent_signal_list):
    i=1
    accent_signal = []
    for i in xrange(len(accent_signal_list)):
        prev_spectrogram = accent_signal_list[i-1]
        spectrogram = accent_signal_list[i]
        val = hwr_fn(np.array(spectrogram)-np.array(prev_spectrogram))
        accent_signal.append(val)
        i += 1

    return accent_signal

def hwr_fn(array):
    array_filtered = filter(lambda x: x > 0, array)
    return sum(array_filtered)

def perform_fft(buff_data):
    signal = scipy.fftpack.fft(buff_data)
    return np.abs(signal[:len(buff_data)/2+1])

def autocorr(signal):
    result = np.correlate(signal, signal, mode='full')
    return result[result.size/2:]

def find_periodicity(accent_signal):
    autocorr_signal = autocorr(accent_signal)
    index_60bpm, index_180bpm = int(np.floor(temp2index(60))), int(np.ceil(temp2index(180)))
    signal_within_index = autocorr_signal[index_180bpm:index_60bpm]
    tempo_index = np.argmax(signal_within_index) + index_180bpm

    return autocorr_signal, tempo_index

def determine_beat_location(accent_signal, tempo_index):
    beat_indices = []
    first_beat_index = np.argmax(accent_signal[:tempo_index])
    beat_indices.append(first_beat_index-1)
  
    index = first_beat_index + tempo_index

    while index < len(accent_signal):
        start_index = index-10
        end_index = index+10
        if index+10 > len(accent_signal):
            end_index = len(accent_signal)

        next_beat_index = np.argmax(accent_signal[start_index:end_index]) + start_index
        beat_indices.append(next_beat_index-1)
        index += tempo_index

    return np.array(beat_indices)

def plot_graphs(wav, accent_signal, beat_indices):
    pl.title('Accent Signal with Beat Times')
    pl.plot(accent_signal)
    for beat in beat_indices:
        pl.axvline(x=beat, color='g')
    pl.savefig('beattimes.png')

    pl.show()

def write_to_file(beat_time):
    beat_time.tofile('beat_time.csv', sep=',')

def temp2index(tempo):
    L = 0.0116
    return 60.0/(tempo*L)

def index2time(index):
    return np.array(index)*0.0116

if __name__ == "__main__":
    wav = read_wav_file('Moskau.wav')
    accent_signal = create_buffer_matrix(wav)
    autocorr_signal, tempo_index = find_periodicity(accent_signal)
    beat_indices = determine_beat_location(accent_signal, tempo_index)
    time_index = index2time(beat_indices)
    plot_graphs(wav, accent_signal, beat_indices)
    write_to_file(time_index)
