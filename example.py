import numpy as np
import wave
from numpy.fft import fft
from numpy.fft import fftfreq
from numpy.fft import ifft
import simpleaudio as sa
import matplotlib.pyplot as plt
import track_class
from scipy.interpolate import interp1d

# YOU SHOULD LOAD FILES WITH 44100HZ IF THATS WHAT IN THE CLASS INIT

fuzz_gain=6.
preamp_gain = 1.0
poweramp_gain = 1.4

# apply EQ and cab

JM = track_class.track_class('raw/robin_JM.wav',44100)

# JM.play_audio() # plays dry input file
#JM.apply_clipping(JM.clip.fuzz_input,JM.clip.fuzz_output,fuzz_gain)

#JM.play_audio()


#JM.plot_basscut() # plots basscut
JM.apply_basscut() # simple bass cut

# JM.play_audio() # plays dry input file

#preamp clipping
#JM.sigmoid_clip(2.2,0.02) # first argument is preamp headroom, higher is cleaner
JM.apply_clipping(JM.clip.tube_input,JM.clip.tube_output,preamp_gain)


#fender amp eq
#JM.plot_cubic_eq(JM.EQ.fender_467_dB,JM.EQ.fender_467_Hz) # can plot amp eq
JM.apply_cubic_eq(JM.EQ.fender_467_dB,JM.EQ.fender_467_Hz)
#JM.apply_fender_eq() # old fender eq

#alternative marshall amp
#JM.plot_cubic_eq(JM_clean.EQ.marshall_387_dB,JM_clean.EQ.marshall_387_Hz) # can plot amp eq
#JM.apply_cubic_eq(JM.EQ.marshall_387_dB,JM.EQ.marshall_387_Hz)
#JM.apply_marshall_eq() # old marshall eq

#power amp clipping
#JM.sigmoid_clip(0.45,0.0) # first argument is poweramp headroom, higher is cleaner
JM.apply_clipping(JM.clip.tube_input,JM.clip.tube_output,poweramp_gain)

#JM.apply_basscut() # simple bass cut

#JM_.apply_CV30_cab_sim() # old cabsim
#JM.plot_cubic_eq(JM.EQ.CV30_dB,JM.EQ.CV30_Hz) # better cabsim
JM.apply_cubic_eq(JM.EQ.CV30_dB,JM.EQ.CV30_Hz) # better cabsim


JM.plot_waveform_vs_raw() # doesnt work

JM.play_audio() #plays processed file


