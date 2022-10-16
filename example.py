import numpy as np
import wave
from numpy.fft import fft
from numpy.fft import fftfreq
from numpy.fft import ifft
import simpleaudio as sa
import matplotlib.pyplot as plt
import track_class

# YOU SHOULD LOAD FILES WITH 44100HZ IF THATS WHAT IN THE CLASS INIT





# apply EQ and cab

JM_clean = track_class.track_class('raw/robin_JM.wav',44100)


# plots eq curves - these are hard coded into the track_class object
JM_clean.plot_basscut()
JM_clean.plot_fender_eq()
JM_clean.plot_CV30_cab_sim()

JM_clean.play_audio() # plays dry input file

JM_clean.apply_fender_eq()

JM_clean.apply_CV30_cab_sim()

JM_clean.play_audio() # plays EQ->cab



# # new track and instead now apply basscut, preamp clipping, EQ, power amp clipping and speaker cab.

JM = track_class.track_class('raw/robin_JM.wav',44100)

JM.apply_basscut()

JM.sigmoid_clip(2.2,0.02) # first argument is preamp headroom, higher is cleaner

JM.apply_fender_eq()

JM.sigmoid_clip(0.65,0.0) # first argument is poweramp headroom, higher is cleaner

JM.apply_CV30_cab_sim()

JM.play_audio()

# JM.plot_waveform_vs_raw()

# high gain example
# new track and instead now apply basscut, preamp clipping, EQ, power amp clipping and speaker cab. 

BIB = track_class.track_class('raw/robin_BIB.wav',44100)

BIB.apply_basscut()

BIB.sigmoid_clip(0.3,0.02) # first argument is preamp headroom, higher is cleaner

BIB.apply_fender_eq()

BIB.blend_clean(0.6)

BIB.sigmoid_clip(0.13,0.0) # first argument is poweramp headroom, higher is cleaner

BIB.apply_CV30_cab_sim()

BIB.play_audio()

#JM.plot_waveform_vs_raw()
