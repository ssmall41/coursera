import math
import numpy as np
from scipy.fft import fft, fftfreq
import soundfile as sf


def freq_to_note(frequency):
    """
        Convert a frequency in Hz to the closest musical note.
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")

    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    A4 = 440.0
    
    # Number of semitones away from A4 (can be fractional)
    semitones_from_a4 = 12 * math.log2(frequency / A4)
    nearest_semitone = round(semitones_from_a4)
    
    # MIDI note number: A4 = 69
    midi_number = 69 + nearest_semitone
    
    # Octave and note index (MIDI: C-1 = 0, so octave = midi//12 - 1)
    octave = midi_number // 12 - 1
    note_index = midi_number % 12
    note_name = note_names[note_index]

    return f"{note_name}{octave}"


def find_dominate_frequency(signal, samplerate):
    """
        Given a time series and the sample rate (samples per second), find the dominate frequency.
    """
    # Setup
    fs = samplerate    
    N = signal.shape[0]  # number of samples
    T = N / fs  # duration in seconds
    t = np.linspace(0, T, N, endpoint=False)
    
    # Compute the DFT
    X = fft(signal)
    freqs = fftfreq(N, d=1/fs)   # corresponding frequency bins

    # Find the dominate frequency
    max_freq_idx = np.argmax(np.abs(X[:(N//2)]))  # Note: it's real valued, so only do half
    max_freq = freqs[max_freq_idx]
    return max_freq


# Read in data  #############################
notes = ["C5", "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5"]

data_flute_nonvib = []
samplerate_flute_nonvib = []
for n in notes:
    data, samplerate = sf.read(f'../data/Flute.nonvib.ff.{n}.stereo.aif')
    data_flute_nonvib.append(data)
    samplerate_flute_nonvib.append(samplerate)

data_oboe = []
samplerate_oboe = []
for n in notes:
    data, samplerate = sf.read(f'../data/Oboe.ff.{n}.stereo.aif')
    data_oboe.append(data)
    samplerate_oboe.append(samplerate)

data_trumpet_nonvib = []
samplerate_trumpet_nonvib = []
for n in notes:
    data, samplerate = sf.read(f'../data/Trumpet.novib.ff.{n}.stereo.aif')
    data_trumpet_nonvib.append(data)
    samplerate_trumpet_nonvib.append(samplerate)

# Iterate over the data, and find the estimated notes #######################
print("Scanning the Flute (no vibrato) data........................")
for note, data, samplerate in zip(notes, data_flute_nonvib, samplerate_flute_nonvib):
    estimated_freq = find_dominate_frequency(data[5000:6000, 0], samplerate)
    estimated_note = freq_to_note(estimated_freq)
    print(f"For actual note {note}, estimating {estimated_note}.")

print("Scanning the Oboe data........................")
for note, data, samplerate in zip(notes, data_oboe, samplerate_oboe):
    estimated_freq = find_dominate_frequency(data[5000:6000, 0], samplerate)
    estimated_note = freq_to_note(estimated_freq)
    print(f"For actual note {note}, estimating {estimated_note}.")

print("Scanning the Trumpet (no vibrato) data........................")
for note, data, samplerate in zip(notes, data_trumpet_nonvib, samplerate_trumpet_nonvib):
    estimated_freq = find_dominate_frequency(data[5000:6000, 0], samplerate)
    estimated_note = freq_to_note(estimated_freq)
    print(f"For actual note {note}, estimating {estimated_note}.")

