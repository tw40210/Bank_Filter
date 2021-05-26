pyequalizer
=======================

This repo is a python equalizer which is implemented by scipy.
##Install
You can install this package by pip directly.
```
pip install pyequalizer
```

And, the following code is the main one you will use in this package. 
```
import pyequalizer.Filter as Filter
type_f = "parabola"
input_file="data/bass.wav"
y, sr = librosa.load(input_file, sr=44100)
y = Filter.equalizer(y, sr, start_iv=200, end_iv=8000, num_iv=100, peak_scale=20, peak_iv = 20, type=type_f)
```
where **start_iv, end_iv** is the frequencys equalizer starts to work and ends;
**num_iv** is the number how many interval will be in. Generally, it means how smooth the equalize curve is.
**peak_scale** is how much will the signal be enhanced.
**peak_iv** is the interval where the peak of equalizing curve is
**type** means the equalizing curve shape. We currently provide "parabola" and "linear".

##Demo
By the mentioned code, we enhance the frequency between 200-8000. In the following figure, it looks like we supress other bands because of librosa spectrum display function.

<img src="demo_result.png" width="100%">
Top one is the original spectrum and bottom one is processed spectrum.


License
-------

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any means.

