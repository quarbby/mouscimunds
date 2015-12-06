#!/usr/bin/env python

#
# Sample code for assignment 5 - MFCC
# CS4347: Sound and Music Computing
#
# Author: DUAN Zhiyan <zhiyan@u.nus.edu>
#

import math
import numpy
import scipy.io.wavfile
import pylab

import scipy.signal
import scipy.fftpack

def M(f):
    """ Translate frequency `f` to mel-scale
    """
    return 1127 * numpy.log(1 + f/700.0)

def Mi(m):
    """ Translate mel-scale `m` to frequency
    """
    return 700 * (numpy.exp(m/1127.0) - 1)

def freq2bin(f, srate, N):
    """ Translate frequency `f` to its corresponding `N`-point FFT bin

    f : frequency
    srate : sampling rate
    N : N-point fft
    """
    return (f / float(srate)) * N

def bin2freq(b, srate, N):
    """ Translate `N`-point FFT bin `b` to its corresponding freqency

    b : frequency bin
    srate : sampling rate
    N : N-point fft
    """
    return (b / float(N)) * float(srate)

def filterbank(min_freq, max_freq, number, srate, N):
    """ Calculate the mel-frequency filterbank.

    min_freq : min frequency
    max_freq : max frequency
    number : the number of filters in the filterbank
    srate : sampling rate
    N : N-point fft
    """
    points = numpy.linspace(M(min_freq), M(max_freq), number + 2)
    freqs = Mi(points)
    bins = freq2bin(freqs, srate, N)

    filters = numpy.zeros((number, N/2 +1))

    for i in xrange(0, number):
        bot = int(math.floor(bins[i]))
        mid = int(round(bins[i+1]))
        top = int(math.ceil(bins[i+2]))

        filters[i][bot:mid] = numpy.linspace(0, 1, mid - bot +1)[:-1]
        filters[i][mid:top+1] = numpy.linspace(1, 0, top - mid +1)

    return filters

def emphasis(signal):
    """ Apply pre-emphasis filter to `signal`

    y(t) = x(t) - 0.95 * x(t-1)
    """
    emphed = signal[1:] - (0.95 * signal[:-1])
    return numpy.insert(emphed, 0, signal[0])

def bucketize(signal, windowsize, overlap):
    """ Slice `signal` into buckets of size `windowsize` with overlap `overlap`
    Calculate the magnitude spectrum and discard the negative frequencies.

    windowsize, overlap : in samples
    """
    bucket_count = len(signal) / (windowsize - overlap) -1
    buckets = numpy.zeros((bucket_count, windowsize/2 + 1))
    hamming = numpy.hamming(windowsize)

    step = windowsize - overlap
    for i in xrange(bucket_count):
        start = i * step
        windowed = emphasis(signal[start:start+windowsize]) * hamming
        buckets[i] = numpy.abs(scipy.fftpack.fft(windowed)[:windowsize/2 +1])

    return buckets

def mfcc(path, windowsize, overlap, M):
    """ Process file stored at `path` and calculate buffer-based MFCC features

    path : wav file path
    windowsize : buffer size, in samples
    overlap : overlap size, in samples
    M : number of mel filters to use

    return: MFCC features for all buffers
    """
    srate, data = scipy.io.wavfile.read(path)

    bank = filterbank(0, srate/2, M, srate, windowsize)
    buckets = bucketize(data/32768.0, windowsize, overlap)
    energies = buckets.dot(bank.transpose())

    return scipy.fftpack.dct(numpy.log10(energies))

def arff(features, path):
    """ Write MFCC `features` to file `path`
    """
    out = open(path, 'w')

    # Header
    out.write("@RELATION music_speech\n")
    for i in range(features.shape[1]-1):
        out.write("@ATTRIBUTE MFCC_%i NUMERIC\n" % i)
    out.write("@ATTRIBUTE class {music,speech}\n\n@DATA\n")

    # Data
    for mfcc in features:
        for i in xrange(len(mfcc)-1):
            out.write("%f," % mfcc[i])
        out.write("%s\n" % ('music' if mfcc[-1] == 1 else 'speech'))

    out.close()

def plot_filterbank(srate, windowsize, filter_count):
    bank = filterbank(0, srate/2, filter_count, srate, windowsize)
    freqs = bin2freq(numpy.arange(0, windowsize/2 +1), srate, windowsize)

    pylab.figure(0)
    pylab.title('26 Triangular MFCC filters, 22050Hz signal, window size 1024')
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Amplitude')
    for filter in bank:
        pylab.plot(freqs, filter, 'o-')

    pylab.xlim(0, 300)
    pylab.savefig('bank-2.png')

    pylab.figure(1)
    pylab.title('26 Triangular MFCC filters, 22050Hz signal, window size 1024')
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Amplitude')
    for filter in bank:
        pylab.plot(freqs, filter, '-')
    pylab.savefig('bank-1.png')

def main():
    M = 26 # number of mel filters
    srate, windowsize, overlap = 22050, 1024, 512

    # Plotting
    plot_filterbank(srate, windowsize, M)

    # Number crunching
    lines = open('music_speech.mf').readlines()
    features = numpy.zeros((len(lines), M*2 + 1)) # M means, M stds, 1 label

    for i, line in enumerate(lines):
        print "processing %03d/%d" % (i+1, len(lines))
        wavfile, label = line.rstrip().split("\t")

        mfcc_buffered = mfcc(wavfile, windowsize, overlap, M)

        features[i][:M] = mfcc_buffered.mean(axis = 0)
        features[i][M:-1] = mfcc_buffered.std(axis = 0)

        features[i][-1] = 1 if label == 'music' else 0

    # Saving results
    arff(features, 'mfcc.arff')

if __name__ == "__main__":
    main()
