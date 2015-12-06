# CS4347 Assignment 3 Machine Learning and Buffer-based Time Domain Audio

import numpy as np
import scipy
from scipy.io import wavfile
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
    print count 

def create_buffer_matrix(wav_float, label):
    buff_length = 1024
    features_matrix = []
    num_buffers = len(wav_float/buff_length)
    
    start = 0
    hopsize = buff_length/2

    for i in xrange(num_buffers):
        end = start + buff_length
        buff_data = wav_float[start:end]
        start  += hopsize

        if not len(buff_data) < buff_length:
            features_matrix.append(calculate_features(buff_data))

    std_dev = calculate_std_dev(features_matrix)
    mean = calculate_mean(features_matrix)
    buff_matrix = []
    buff_matrix = buff_matrix + mean
    buff_matrix = buff_matrix + std_dev
    buff_matrix.append(label)

    return buff_matrix

def calculate_features(buff_data):
    RMS = calculate_rms(buff_data)
    PAR = calculate_par(buff_data, RMS)
    ZCR = calculate_zcr(buff_data)
    MAD = calculate_mad(buff_data)
    MEANAD = calculate_meanad(buff_data)

    return [RMS, PAR, ZCR, MAD, MEANAD]

def calculate_std_dev(features_matrix):
    features_matrix = np.array(features_matrix)
    rms_std_dev = np.std(features_matrix[:,0])
    par_std_dev = np.std(features_matrix[:,1])
    zcr_std_dev = np.std(features_matrix[:,2])
    mad_std_dev = np.std(features_matrix[:,3])
    meanad_st_dev = np.std(features_matrix[:,4])

    return [rms_std_dev, par_std_dev, zcr_std_dev, mad_std_dev, meanad_st_dev]

def calculate_mean(features_matrix):
    features_matrix = np.array(features_matrix)
    rms_mean = np.mean(features_matrix[:,0])
    par_mean = np.mean(features_matrix[:,1])
    zcr_mean = np.mean(features_matrix[:,2])
    mad_mean = np.mean(features_matrix[:,3])
    meanad_mean = np.mean(features_matrix[:,4])

    return [rms_mean, par_mean, zcr_mean, mad_mean, meanad_mean]

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

def calculate_meanad(wav):
    mean = np.mean(wav)    
    return np.mean(abs(wav-mean))

def write_header(file):
    file.write("@RELATION music_speech\n")
    file.write("@ATTRIBUTE RMS_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE PAR_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE ZCR_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE MAD_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE MEAN_AD_MEAN NUMERIC\n")
    file.write("@ATTRIBUTE RMS_STD NUMERIC\n")
    file.write("@ATTRIBUTE PAR_STD NUMERIC\n")
    file.write("@ATTRIBUTE ZCR_STD NUMERIC\n")
    file.write("@ATTRIBUTE MAD_STD NUMERIC\n")
    file.write("@ATTRIBUTE MEAN_AD_STD NUMERIC\n")
    file.write("@ATTRIBUTE class {music,speech}\n\n")
    file.write("@DATA\n")

if __name__ == "__main__":
    file = open('buffbased.arff', 'w')
    write_header(file)
    run("./music_speech.mf", file)
    file.close()
