import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from Filter import band_filter, high_filter, low_filter, equalizer
from utils import smooth_interval_scale, curve_maker, curve_calcu

def aaa():
    print(2)
    return 2

if __name__ == '__main__':
    # import pyximport; pyximport.install()
    # import Ceq_pck.Filter_cy as CFilter
    # import Cython_pck.fib as fib


    a, b = smooth_interval_scale(200, 600, 3, 10, 1, "parabola")

    interval = np.arange(200, 600 + (600 - 200) / 10, (600 - 200) / 10)
    para = curve_maker([10, 1], [5, 5], "parabola")
    y = curve_calcu(8, para, "parabola")

    input_file = "goodbye.wav"
    hop_length = 1024
    interval = [200, 400, 600]
    scale = [1, 10, 20, 1]
    y, sr = librosa.load(input_file, sr=44100)
    y = y[:int(sr * 20)]
    plt.figure()
    bef_stft = np.abs(librosa.stft(y))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                             x_axis='time')

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    type_f = "parabola"
    y1 = CFilter.equalizer(y, sr, start_iv=200, end_iv=8000, num_iv=100, peak_scale=20, peak_iv=20, type=type_f)
    y = equalizer(y, sr, start_iv=200, end_iv=8000, num_iv=100, peak_scale=20, peak_iv = 20, type=type_f)
    aft_stft = np.abs(librosa.stft(y))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                             x_axis='time')

    plt.tight_layout()
    plt.show()
    # "linear", "parabola"
    librosa.output.write_wav(f"output_{type_f}.wav", y, sr=sr, norm=True)
