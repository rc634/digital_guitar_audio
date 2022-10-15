import numpy as np
from numpy.fft import fft
from numpy.fft import fftfreq
from numpy.fft import ifft
import simpleaudio as sa
import matplotlib.pyplot as plt

def hardclip(audio,headroom):
	ups = np.where(audio > headroom)
	downs = np.where(audio < -headroom)
	audio[ups[0]] = headroom
	audio[downs[0]] = -headroom

def cubicclip(audio,threshold):
	threshold *= np.max(np.abs(output))
	audio -= ((audio/threshold)**3)*threshold


sample_rate = 44100
time=5
#t=np.linspace(0,time,num=time*sample_rate)

E2 = 82.41*2*np.pi
B2 = 123.47*2*np.pi
E3 = E2*2

D3 = 146.83*2*np.pi
A3 = 220*2*np.pi
D4 = D3*2

A2 = 110*2*np.pi


bib_wav = sa.WaveObject.from_wave_file('raw/robin_BIB.wav')
bib_play = bib_wav.play()
bib_play.wait_done()

onesec = np.linspace(0,1,num=sample_rate)

pcE = (np.sin(onesec*E2)+np.sin(onesec*B2)+np.sin(onesec*E3))*np.exp(-onesec)
pcD = (np.sin(onesec*D3)+np.sin(onesec*A3)+np.sin(onesec*D4))*np.exp(-onesec)
pcA = (np.sin(onesec*A2)+np.sin(onesec*E3)+np.sin(onesec*A3))*np.exp(-onesec)

output = np.concatenate((np.zeros(sample_rate),pcE,np.zeros(sample_rate),pcD,np.zeros(sample_rate),pcA))


output *= 32767 / np.max(np.abs(output))
hardclip(output,10000)

output = output.astype(np.int16)

play = sa.play_buffer(output,1,2,sample_rate)

play.wait_done()
play.stop()

plt.plot(output)
plt.show()

spectrograph = fft(output)
freq = sample_rate*fftfreq(output.shape[-1])
plt.plot(freq,np.abs(spectrograph))
plt.xlim(0,20000)

plt.show()



#############
# apply eq filter
##############

eqfilter = np.exp(-np.abs(freq**2)/(400**2))
filtered_spectrum = np.multiply(spectrograph,eqfilter)
plt.plot(freq,eqfilter)
plt.show()

pp_signal = ifft(filtered_spectrum)
pp_signal = np.array(pp_signal.real)
plt.plot(pp_signal.real)
plt.show()



pp_signal = np.array(pp_signal)
print(pp_signal)
#pp_signal *= 32767 / np.max(np.abs(pp_signal))
pp_signal = pp_signal.astype(np.int16)
play = sa.play_buffer(pp_signal,1,2,sample_rate)

play.wait_done()
play.stop()


