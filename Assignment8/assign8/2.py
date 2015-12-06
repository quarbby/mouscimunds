# CS4347 Assignment 8: Synthesis Beyond 2 Sine Waves Part 2

import numpy as np

import scipy

import pylab

fs = 44100
length = 1.0

def get_perfect_sine_wave(freq):
    time = np.arange(0.0, 1.0, 1./fs)
    sine_wave = np.sin(2.0 * np.pi * freq *  time)
    return sine_wave

def create_lookup_table(num_samples):
    time = np.linspace(0, 1.0, num_samples)
    sine_wave = np.sin(2.0 * np.pi * time)
    return sine_wave

def get_wave_no_interpolation(sine_wave, freq, lut_size):
    index_to_advance = freq/fs * lut_size

    wave = np.zeros(fs)
    phase = 0.0
    for i in xrange(fs):
        phase_int = int(phase)
        wave[i] = sine_wave[phase_int]
        phase += index_to_advance

        # Handle wrap around
        if (phase >= lut_size):
            phase = phase % lut_size

    return wave

def get_wav_interpolation(sine_wave, freq, lut_size):
    index_to_advance = freq/fs * lut_size

    wave = np.zeros(fs)
    phase = 0.0

    for i in xrange(fs):
        x0 = np.floor(phase)
        x1 = x0 + 1

        if (x1 >= lut_size):
            x1 = x1 % length

        y0 = sine_wave[x0]
        y1 = sine_wave[x1]
        y = y0 + (y1-y0)*(phase-x0)/(x1-x0)

        wave[i] = y

        phase += index_to_advance

        if (phase >= lut_size):
            phase = phase % lut_size

    return wave

def max_error(perfect_sine_wave, lut_sine_wave):
    max_error = np.max(np.abs(lut_sine_wave - perfect_sine_wave))
    max_audio_file_error = 32767 * max_error

    return round(max_audio_file_error,3)

def write_file(err_1, err_2, err_3, err_4, err_5, err_6, err_7, err_8):
    out_file = open("output.txt", "w")
    out_file.write("Frequency\tInterpolation\t16384-sample\t2048-sample\n")
    
    out = "100Hz\t\tNo\t\t" + str(err_1) + "\t" + str(err_2) + "\n"
    out += "\t\tLinear\t\t" + str(err_3) + "\t" +  str(err_4) + "\n"
    out += "1234.56Hz\tNo\t\t" + str(err_5) + "\t" + str(err_6) + "\n"
    out += "\t\tLinear\t\t" + str(err_7) + "\t" + str(err_8) + "\n"

    out_file.write(out)

    out_file.close()

if __name__ == '__main__':
    freq1 = 100.0
    freq2 = 1234.56

    # Get perfect sine waves
    perfect_sine_wave_f1 = get_perfect_sine_wave(freq1)
    perfect_sine_wave_f2 = get_perfect_sine_wave(freq2)

    # Generate the first LUT
    lut_size_1 = 16384
    sin_wave = create_lookup_table(lut_size_1)
    sin_f1_no_interpolation = get_wave_no_interpolation(sin_wave, freq1, lut_size_1)
    sin_f1_interpolation = get_wav_interpolation(sin_wave, freq1, lut_size_1)
    sin_f2_no_interpolation = get_wave_no_interpolation(sin_wave, freq2, lut_size_1)
    sin_f2_interpolation = get_wav_interpolation(sin_wave, freq2, lut_size_1)

    '''
    pl.plot(perfect_sine_wave_f1, color='b', label='perfect')
    pl.plot(sin_f1_no_interpolation, color='g', label='no interpolation')
    pl.plot(sin_f1_interpolation, color='r', label='interpolation')
    pl.legend()
    pl.show()
    '''

    # Generate the second LUT
    lut_size_2 = 2048
    sine_wave_2 = create_lookup_table(lut_size_2)
    sin2_f1_no_interpol = get_wave_no_interpolation(sine_wave_2, freq1, lut_size_2)
    sin2_f1_interpol = get_wav_interpolation(sine_wave_2, freq1, lut_size_2)
    sin2_f2_no_interpol = get_wave_no_interpolation(sine_wave_2, freq2, lut_size_2)
    sin2_f2_interpol = get_wav_interpolation(sine_wave_2, freq2, lut_size_2)

    # Get errors
    err_1 = max_error(perfect_sine_wave_f1, sin_f1_no_interpolation)
    err_2 = max_error(perfect_sine_wave_f1, sin2_f1_no_interpol)
    err_3 = max_error(perfect_sine_wave_f1, sin_f1_interpolation)
    err_4 = max_error(perfect_sine_wave_f1, sin2_f1_interpol)
    err_5 = max_error(perfect_sine_wave_f2, sin_f2_no_interpolation)
    err_6 = max_error(perfect_sine_wave_f2, sin2_f2_no_interpol)
    err_7 = max_error(perfect_sine_wave_f2, sin_f2_interpolation)
    err_8 = max_error(perfect_sine_wave_f2, sin2_f2_interpol)

    write_file(err_1, err_2, err_3, err_4, err_5, err_6, err_7, err_8)
