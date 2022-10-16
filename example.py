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

x=np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
y=np.array([0.,1.,1.,0.5,0.7,1.,1.3,1.6,1.7,1.7,1.7])


# apply EQ and cab

JM = track_class.track_class('raw/robin_JM.wav',44100)



JM.play_audio() # plays dry input file

#JM.plot_basscut() # plots basscut
JM.apply_basscut() # simple bass cut

#preamp clipping
#JM.sigmoid_clip(2.2,0.02) # first argument is preamp headroom, higher is cleaner

#JM.plot_cubic_eq(JM_clean.EQ.fender_467_dB,JM_clean.EQ.fender_467_Hz) # can plot amp eq
#JM.apply_cubic_eq(JM.EQ.fender_467_dB,JM.EQ.fender_467_Hz)
#JM.apply_fender_eq() # old fender eq

#alternative marshall amp
JM.apply_cubic_eq(JM.EQ.marshall_387_dB,JM.EQ.marshall_387_Hz)

#power amp clipping
JM.sigmoid_clip(0.45,0.0) # first argument is poweramp headroom, higher is cleaner

#JM_.apply_CV30_cab_sim() # old cabsim
JM.apply_cubic_eq(JM.EQ.CV30_dB,JM.EQ.CV30_Hz) # better cabsim

JM.play_audio() #plays processed file


