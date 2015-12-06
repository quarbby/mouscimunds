# Assignment 5: Perceptual Audio Features

import numpy as np
import math

import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft, dct

import pylab as pl

import mel 

def run(filename, file):
    ground_truth_file = open(filename, 'r')
    mel_filter_bank = mel.mel_filter()

    for line in ground_truth_file:
        line_split = line.strip().split("\t")

        wav = scipy.io.wavfile.read(line_split[0])
        wav_float = wav[1]/32768.0

        buff_matrix = create_buffer_matrix(wav_float, line_split[1], mel_filter_bank)
        write_data(buff_matrix, file)

def write_data(buff_matrix, file):
    for group in buff_matrix:
        line = ",".join(map(str,group))
        file.write(line + "\n")

def create_buffer_matrix(wav_float, label, mel_filter_bank):
    buff_length = 1024.0
    num_buffers = int(len(wav_float)/buff_length)

    start = 0
    hopsize = buff_length/2

    buff_matrix = []

    for i in xrange(num_buffers):
        end = start + buff_length
        buff_data = np.array(wav_float[start:end])
        start  += hopsize

        if not len(buff_data) < buff_length:
            mfcc = calculate_mfcc(buff_data, mel_filter_bank)
            features = calculate_features(mfcc)
            features.append(label)

        buff_matrix.append(features)

    return buff_matrix

def calculate_mfcc(buff_data, mel_filter_bank): 
    buff_data = perform_pre_emp(buff_data)
    buff_data = multiply_hamming_window(buff_data)

    fft_data = perform_fft(buff_data)
    x_freq_range = np.arange(0, len(fft_data))
    x_freq_range = map(bin_to_freq, x_freq_range)
   
    #pl.figure()
    #pl.plot(x_range, fft_data)

    filter_data = get_dot_products(fft_data, mel_filter_bank)
    log_data = np.log10(filter_data)
    mfcc_vector = perform_dct(log_data)

    return mfcc_vector

def get_dot_products(buff_data, mel_filter_bank):
    dot_products = []

    for i in xrange(len(mel_filter_bank)):
        [filter_start, filter_end, filter_data] = mel_filter_bank[i]

        bin_filter_start, bin_filter_end = freq_to_bin(filter_start), freq_to_bin(filter_end)
        bin_filter_center = np.round((bin_filter_end-bin_filter_start)/2) + bin_filter_start
        
        y_range_1 = np.linspace(0.0, 1.0, num=bin_filter_center - bin_filter_start)
        y_range_2 = np.linspace(1.0, 0.0, num=bin_filter_end - bin_filter_center)
        y_range = np.append(y_range_1, y_range_2)

        front_pad = np.zeros(bin_filter_start)
        back_pad = np.zeros(513-bin_filter_end)

        filter_padded = np.concatenate((front_pad, y_range, back_pad))

        dot_filter = np.dot(buff_data, filter_padded)
        dot_products.append(dot_filter)

    return dot_products

def bin_to_freq(bin):
    fs = 22050
    window_size = 1024

    return bin * (fs/ window_size) 

def freq_to_bin(freq):
    fs = 22050.
    window_size = 1024
    bin_size = fs/window_size
    
    bin_num = freq/bin_size

    return int(bin_num)

def perform_pre_emp(buff_data):
    buff_data = list(buff_data)
    buff_data_shifted = [0] + buff_data
    buff_data_shifted = buff_data_shifted[:-1]
    return list(np.array(buff_data) - 0.95*np.array(buff_data_shifted))

def multiply_hamming_window(buff_data):
    window = scipy.signal.hamming(len(buff_data))
    return [a*b for a,b in zip(list(window), list(buff_data))]

def perform_fft(buff_data):
    signal = scipy.fftpack.fft(buff_data)

    return np.abs(signal[:len(buff_data)/2+1])

def perform_dct(data):
    return scipy.fftpack.dct(data)

def calculate_features(mfcc):
    return [np.mean(mfcc), np.std(mfcc)]

def write_header(file):
    file.write("@RELATION music_speech\n")
    file.write("@ATTRIBUTE MFCC_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE MFCC_STD_DEV NUMERIC\n")
    file.write("@ATTRIBUTE class {music,speech}\n\n")
    file.write("@DATA\n")

if __name__ == "__main__":
    file = open('audiofeatures.arff', 'w')
    write_header(file)
    run("./music_speech.mf", file)
    file.close()
