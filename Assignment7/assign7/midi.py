# CS4347 Assignment 7: Audio Synthesis with Sine Waves

import numpy as np

import scipy
from scipy.io import wavfile
from scipy.fftpack import fft 
from scipy import signal

import pylab as pl
import matplotlib


midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]

length_of_note = 0.25
num_bits = 16
sampling_rate = 32000
num_sine_wavs = 4

time = np.arange(0, length_of_note, 1./sampling_rate)

def play_midis():
    notes = []
    for midi in midis:
        note = []
        if midi == 0:
            note = np.zeros(len(time))
        else: 
           fun_freq = calc_fundamental_freq(midi) 
           note = harmonic_additive_syn(fun_freq)

        notes.append(list(note))
    return notes

def calc_fundamental_freq(m):
    return 440*np.power(2, (m-69.)/12)

def harmonic_additive_syn(fun_freq):
    waves = np.sin(2 * np.pi * fun_freq * time)

    for i in xrange(num_sine_wavs):
        wave = 0.5 * np.sin(2.0 * np.pi * (i+1) * fun_freq * time)
        waves = waves + wave

    return waves
    #return np.sin(2*np.pi * fun_freq*time)

def write_audio(notes, filename):
    audio = list(np.ravel(notes))
    scaled = np.int16(audio/np.max(np.abs(audio))*32767)
    
    scipy.io.wavfile.write(filename, sampling_rate, scaled)
    return np.array(audio)/32768.0

def db_spectrum(time_domain_data, window):
    windowed_signal = time_domain_data * window
    fft = scipy.fftpack.fft(windowed_signal)
    fft = fft[:len(fft)/2+1]
    magfft = np.abs(fft)

    epsilon = pow(10, -10)
    db = 20 * np.log10(magfft + epsilon)

    return db

def save_spectrogram(notes, filename):
    window_size = 512
    overlap = 512/2

    num_buffers = int(len(notes)/window_size)*2
    window = scipy.signal.blackman(window_size)

    data = []
    for i in xrange(num_buffers):
        start = i * overlap
        end = start + window_size

        note_data = notes[start:end]

        if len(note_data) < window_size:
            break

        db = db_spectrum(note_data, window)
        
        data.append(db)

    data = np.transpose(np.array(data))

    pl.imshow(data, origin='lower', aspect='auto', interpolation='nearest')
    pl.suptitle(filename)
    pl.ylabel('Frequency (log)')
    pl.xlabel('Time (bins)')

    pl.savefig(filename)
    pl.show()


def apply_adsr(notes):
    phase_length = len(notes[0])/5
   
    amp = 100.0
    attack = np.linspace(0, amp, phase_length)
    decay = np.linspace(amp, amp/2, phase_length)
    sustain = np.linspace(amp/2, amp/2, phase_length*2)
    release = np.linspace(amp/2, 0, phase_length)
    
    envelope = np.concatenate((attack, decay, sustain, release))

    envelope_notes = []

    for note in notes:
        env_note = np.array(note*envelope)
        envelope_notes.append(env_note)

    return envelope_notes

if __name__ == '__main__':
    notes = play_midis()
    notes_concat = write_audio(notes, 'notes.wav')
    save_spectrogram(notes_concat, 'spectrogram-notes.png')

    envelope = apply_adsr(notes)
    envelop_concat = write_audio(envelope, 'notes-adsr.wav')
    save_spectrogram(envelop_concat, 'spectrogram-notes-adsr.png')

