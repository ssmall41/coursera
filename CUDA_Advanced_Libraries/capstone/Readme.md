# Note Detector

This program takes snippets of musical instruments playing a note and attempts to identify that note and octave. The program takes an input snippet, transforms it using a DFT, identifies the dominant frequency, and finds the closest note.

The source code hardcodes several instruments (namely, Flute, Oboe, and Trumpet) and assumes data in the format provided by the [Musical Instrument Samples Database](https://theremin.music.uiowa.edu/index.html) provided by the University of Iowa Electronic Music Studios. The code also uses the 5th octave by default, but this can be changed in code.

## Required Data
Before running, input data is needed and expected to be provided in the directory `data`. To use the code as-is, data for the [Flute (non-vibrato)](https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISFlute2012.html), [Oboe](https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISOboe2012.html), and [Trumpet (non-vibrato)](https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbTrumpet2012.html) are required. Simply download the `.zip` files for each instrument and unzip them into the `data` directory.

## Compiling and Running
To compile, simply run:
```
make all
```

To compile and run, use the included script
```
sh run.sh
```

## Analysis
An example output for the Oboe in the 5th octave is as follows:
```
For actual note C5, estimating C6.
For actual note Db5, estimating Db6.
For actual note D5, estimating D6.
For actual note Eb5, estimating Eb6.
For actual note E5, estimating E6.
For actual note F5, estimating F6.
For actual note Gb5, estimating Gb6.
For actual note G5, estimating G5.
For actual note Ab5, estimating Ab6.
For actual note A5, estimating A6.
For actual note Bb5, estimating Bb5.
For actual note B5, estimating B5.
```

Using an FFT to identify the pitch class seems to be fairly accurate, however the octave can be tricky. In the above output, the pitch class is always correct, but the octave is often wrong. This is consistent across many instruments and is a known "trap". Many instruments do NOT always produce the strongest frequency at the note they play, but rather in different octaves of that note. Identifying the correct note, and more than just the pitch class, requires more than just a simple FFT.

A quick note: there are sometimes errors in the pitch class as well, but they are much rarer.

## Notebook
There is also a directory called `notebooks`. This includes some Python notebooks used for analyzing the data. It's not needed for running the programming.


