# CS4347 Assignment 4 Machine Learning and Spectral Features of Audio

import numpy as np
import math

import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft

import pylab as pl

def run(filename, file):
    ground_truth_file = open(filename, 'r')
    count = 0
    for line in ground_truth_file:
        line_split = line.strip().split("\t")

        wav = scipy.io.wavfile.read(line_split[0])
        wav_float = wav[1]/32768.0

        buff_matrix = create_buffer_matrix(wav_float, line_split[1])
        line = ",".join(map(str,buff_matrix))
        file.write(line + "\n")
        count += 1

def create_buffer_matrix(wav_float, label):
    buff_length = 1024
    features_matrix = []
    num_buffers = len(wav_float/buff_length)
    
    start = 0
    hopsize = buff_length/2

    # Required for Spectral Flux 
    prev_buff = list(np.zeros(buff_length/2+1))

    for i in xrange(num_buffers):
        end = start + buff_length
        buff_data = wav_float[start:end]
        start  += hopsize

        if not len(buff_data) < buff_length:
            buff_data = multiply_hamming_window(buff_data)
            buff_data = perform_dft(buff_data)
            features = calculate_spectral_features(np.abs(buff_data), prev_buff)
            features_matrix.append(features)
            prev_buff = np.abs(buff_data)

    std_dev = calculate_std_dev(features_matrix)
    mean = calculate_mean(features_matrix)
    buff_matrix = []
    buff_matrix = buff_matrix + mean
    buff_matrix = buff_matrix + std_dev
    buff_matrix.append(label)

    return buff_matrix

def multiply_hamming_window(buff_data):
    window = scipy.signal.hamming(len(buff_data))
    return [a*b for a,b in zip(list(window), list(buff_data))]

def perform_dft(buff_data):
    signal = scipy.fftpack.fft(buff_data)
    return signal[:len(buff_data)/2+1]

def calculate_spectral_features(buff_data, prev_buff):
    SC = calculate_sc(buff_data)
    SRO = calculate_sro(buff_data)
    SFM = calculate_sfm(buff_data)
    PAR = calculate_par(buff_data)
    SF = calculate_sf(buff_data, prev_buff)

    return [SC, SRO, SFM, PAR, SF]

def calculate_std_dev(features_matrix):
    features_matrix = np.array(features_matrix)
    sc_std_dev = np.std(features_matrix[:,0])
    sro_std_dev = np.std(features_matrix[:,1])
    sfm_std_dev = np.std(features_matrix[:,2])
    parfft_std_dev = np.std(features_matrix[:,3])
    flux_st_dev = np.std(features_matrix[:,4])

    #print str(sc_std_dev) + " " + str(sro_std_dev) + " " + str(sfm_std_dev) + " " + str(parfft_std_dev) + " " + str(flux_st_dev)

    return [sc_std_dev, sro_std_dev, sfm_std_dev, parfft_std_dev, flux_st_dev]

def calculate_mean(features_matrix):
    features_matrix = np.array(features_matrix)
    sc_mean = np.mean(features_matrix[:,0])
    sro_mean = np.mean(features_matrix[:,1])
    sfm_mean = np.mean(features_matrix[:,2])
    parfft_mean = np.mean(features_matrix[:,3])
    flux_mean = np.mean(features_matrix[:,4])

    #print str(sc_mean) + " " + str(sro_mean) + " " + str(sfm_mean) + " " + str(parfft_mean) + " " + str(flux_mean)

    return [sc_mean, sro_mean, sfm_mean, parfft_mean, flux_mean]

def calculate_sc(wav):
    numerator = wav*np.arange(len(wav))
    return sum(numerator) / sum(wav)

def calculate_sro(wav):
    l_energy = 0.85 * sum(wav)
    cum_sum = np.cumsum(np.array(wav))
    return np.where(cum_sum >= l_energy)[0][0]

def calculate_sfm(wav):
    ln_wav = [np.log(i) for i in wav]
    numerator = math.exp(sum(ln_wav)/len(wav))
    denom = sum(wav)/len(wav)
    return numerator/denom

def calculate_sf(wav, prev_buff):
    wav_array = np.array(wav)
    prev_buff_array = np.array(prev_buff)
    diff = np.array(wav_array - prev_buff_array)
    sign_diff =  np.sign(diff)
    sign_diff = [x if x > 0 else 0 for x in sign_diff]

    return sum(diff*sign_diff)

def calculate_rms(wav):
    return np.sqrt(np.mean(np.square(wav)))

def calculate_par(wav):
    rms = calculate_rms(wav)
    return max(np.absolute(wav))/rms

def write_header(file):
    file.write("@RELATION music_speech\n")
    file.write("@ATTRIBUTE SC_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE SRO_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE SFM_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE PARFFT_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE FLUX_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE SC_STD NUMERIC\n")
    file.write("@ATTRIBUTE SRO_STD NUMERIC\n")
    file.write("@ATTRIBUTE SFM_STD NUMERIC\n")
    file.write("@ATTRIBUTE PARFFT_STD NUMERIC\n")
    file.write("@ATTRIBUTE FLUX_STD NUMERIC\n")
    file.write("@ATTRIBUTE class {music,speech}\n\n")
    file.write("@DATA\n")

if __name__ == "__main__":
    file = open('spectralfeatures.arff', 'w')
    write_header(file)
    run("./music_speech.mf", file)
    file.close()
