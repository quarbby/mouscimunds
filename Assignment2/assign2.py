# CS4347 Assignment 2 Machine Learning with Time Domain Audio
import numpy as np
import scipy
from scipy.io import wavfile
import pylab as pl

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

def calculate_features(ground_truth):
    features = []
    for pair in ground_truth:
        wav = scipy.io.wavfile.read(pair[0])
        wav_float = wav[1]/32768.0
        RMS = calculate_rms(wav_float)
        PAR = calculate_par(wav_float, RMS)
        ZCR = calculate_zcr(wav_float)
        MAD = calculate_mad(wav_float)
        features.append([RMS,PAR,ZCR,MAD,pair[1]])
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

def write_arff_file(features, ground_truth):
    file = open("out.arff", 'w')
    write_header(file)
    
    for group in features:
        line = ",".join(map(str,group))
        file.write(line + "\n")

def write_header(file):
    file.write("@RELATION music_speech\n")
    file.write("@ATTRIBUTE RMS NUMERIC\n")
    file.write("@ATTRIBUTE PAR NUMERIC\n")
    file.write("@ATTRIBUTE ZCR NUMERIC\n")
    file.write("@ATTRIBUTE MAD NUMERIC\n")
    file.write("@ATTRIBUTE class {music,speech}\n\n")
    file.write("@DATA\n")

def plot_graphs(features):
    features_music, features_speech = get_music_speech(features)

    # Plot ZCR vs PAR
    pl.figure()
    pl.scatter(features_music[:,2], features_music[:,1], label="music")
    pl.scatter(features_speech[:,2], features_speech[:,1], color="red", label="speech")
    pl.legend()
    pl.suptitle("ZCR vs PAR", fontsize=14, fontweight="bold")
    pl.xlabel("ZCR", fontsize=14, fontweight="bold")
    pl.ylabel("PAR", fontsize=14, fontweight="bold")
    pl.savefig("zcr_par.png")

    # Plot MAD vs RMS 
    pl.figure()
    pl.scatter(features_music[:,3], features_music[:,0], label="music")
    pl.scatter(features_speech[:,3], features_speech[:,0], color="red", label="speech")
    pl.legend()
    pl.suptitle("MAD vs RMS", fontsize=14, fontweight="bold")
    pl.xlabel("MAD", fontsize=14, fontweight="bold")
    pl.ylabel("RMS", fontsize=14, fontweight="bold")
    pl.savefig("mad_rms.png")

    pl.show()

def get_music_speech(features):
    features_music = []
    features_speech = []

    for group in features:
        if group[-1] == 'music':
            features_music.append(group[:-1])
        elif group[-1] == 'speech':
            features_speech.append(group[:-1])

    features_music = np.array(features_music)
    features_speech = np.array(features_speech)

    return features_music, features_speech

if __name__ == "__main__":
    ground_truth = read_ground_truth_file("./music_speech.mf")
    features = calculate_features(ground_truth)
    #write_to_file(features)
    write_arff_file(features, ground_truth)
    plot_graphs(features)
