# Ng Hui Xian Lynnette  A0119646X
# CS4347 Assignment 1 Time Domain Audio
import numpy as np
import scipy
from scipy.io import wavfile

def read_ground_truth_file(filename):
    ground_truth = []
    file = open(filename, 'r')
    for line in file:
       line_split = line.strip().split("\t")
       ground_truth.append((line_split[0], line_split[1]))
    return ground_truth

def load_wave_files(ground_truth):
    wav_files = []
    for pair in ground_truth:
        wav = scipy.io.wavfile.read(pair[0])
        wav_float = wav[1]/32768.0
        wav_files.append((pair[0], wav_float))
    return wav_files

def calculate_features_2(ground_truth):
    features = []
    for pair in ground_truth:
        wav = scipy.io.wavfile.read(pair[0])
        wav_float = wav[1]/32768.0
        RMS = calculate_rms(wav_float)
        PAR = calculate_par(wav_float, RMS)
        ZCR = calculate_zcr(wav_float)
        MAD = calculate_mad(wav_float)
        features.append([pair[0],RMS,PAR,ZCR,MAD])
    return features

def calculate_features(wav_files):
    features = []
    for pair in wav_files:
        wav = pair[1]
        RMS = calculate_rms(wav)
        PAR = calculate_par(wav, RMS)
        ZCR = calculate_zcr(wav)
        MAD = calculate_mad(wav)
        features.append([pair[0],RMS,PAR,ZCR,MAD])
    return features

def calculate_rms(wav):
    return np.sqrt(np.mean(np.square(wav)))

def calculate_par(wav, rms):
    return max(np.absolute(wav))/rms

def calculate_zcr(wav):
    zcr_sum = len(scipy.where(wav[:-1]*wav[1:]<0)[0])
    return zcr_sum*1.0/(len(wav)-1)

def calculate_mad(wav):
    med = np.median(wav)
    abs_val = np.abs(med-wav)
    return np.median(abs_val)

def write_to_file(features):
    file = open("out.csv", 'w')
    for group in features:
        line = ",".join(map(str,group))
        file.write(line + "\n")

if __name__ == "__main__":
    ground_truth = read_ground_truth_file("./music_speech.mf")
    #wav_files = load_wave_files(ground_truth)
    #features = calculate_features(wav_files)
    features = calculate_features_2(ground_truth)
    write_to_file(features)
