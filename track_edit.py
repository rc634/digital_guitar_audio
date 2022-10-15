import numpy as np
import wave
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
	#threshold *= np.max(np.abs(output))
	audio -= ((audio/threshold)**3)*threshold

def softclip(audio,headroom):
	audio += -audio +  2./(1.+np.exp(-(audio+0.005)/headroom)) -1.1
	#audio = 1./(np.abs(audio) + 0.001)

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


# bib_wav = sa.WaveObject.from_wave_file('raw/robin_JM.wav')
# bib_play = bib_wav.play()
# bib_play.wait_done()
# bib_play.stop()


ifile = wave.open('raw/robin_JM.wav')
samples = ifile.getnframes()
audio = ifile.readframes(samples)
audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)[::2]
audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
max_int16 = 2**15
audio_normalised = audio_as_np_float32 / max_int16
# plt.plot(audio_normalised)
# plt.show()

raw_audio = audio_normalised*32767 / np.max(np.abs(audio_normalised)) * 0.8
raw_audio = raw_audio.astype(np.int16)
play = sa.play_buffer(raw_audio,1,2,sample_rate)
play.wait_done()
play.stop()


#############
# apply eq filter to raw signal
##############
spectrograph = fft(raw_audio)
freq = sample_rate*fftfreq(audio_normalised.shape[-1])

fender_f = [0.,30,40,70,100,400,1000,2000,4000,10000,30000]
fender_gain = [0.,0.5,0.5,0.45,0.4,0.25,0.5,0.75,0.9,1.,1.]

fender_gainfftb = np.zeros(len(freq))
for i in range(len(freq)):
	frq = np.abs(freq[i])
	A = np.abs(fender_f - frq)
	a = np.argmin(A)
	if frq == fender_f[a]:
		fender_gainfftb[i] = frq
	elif frq > fender_f[a]:
		fender_gainfftb[i] = ((fender_f[a+1]-frq)*fender_gain[a]+(frq-fender_f[a])*fender_gain[a+1])/(fender_f[a+1]-fender_f[a])
	else:
		fender_gainfftb[i] = ((fender_f[a]-frq)*fender_gain[a-1]+(frq-fender_f[a-1])*fender_gain[a])/(fender_f[a]-fender_f[a-1])

plt.loglog(fender_f,fender_gain)
plt.loglog(freq,fender_gainfftb)
plt.show()

eqfilter = np.exp(-np.abs(freq**2)/(10000**2))
eqfilter -= np.exp(-np.abs(freq**2)/100**2)
filtered_spectrum = np.multiply(spectrograph,fender_gainfftb)
#filtered_spectrum = np.multiply(spectrograph,eqfilter)


# plt.plot(freq,eqfilter)
# plt.show()

pp_signal = ifft(filtered_spectrum)
pp_signal = np.array(pp_signal.real)

# plt.plot(pp_signal.real)
# plt.show()

pp_signal = np.array(pp_signal)
pp_signal *= 32767 / np.max(np.abs(pp_signal))
pp_signal = pp_signal.astype(np.int16)
play = sa.play_buffer(pp_signal,1,2,sample_rate)

play.wait_done()
play.stop()
######################




exit()



# plt.plot(audio_normalised*32767)
softclip(audio_normalised,0.02)
# plt.plot(audio_normalised*32767)
# plt.show()


audio_normalised *= 32767 / np.max(np.abs(audio_normalised)) * 0.8
audio_normalised = audio_normalised.astype(np.int16)
#plt.plot(audio_normalised)


play = sa.play_buffer(audio_normalised,1,2,sample_rate)
play.wait_done()
play.stop()

spectrograph = fft(audio_normalised)
freq = sample_rate*fftfreq(audio_normalised.shape[-1])

# plt.plot(freq,np.abs(spectrograph))
# plt.xlim(0,20000)
# plt.show()


#############
# apply eq filter
##############

eqfilter = np.exp(-np.abs(freq**2)/(1000**2))
eqfilter -= np.exp(-np.abs(freq**2)/100**2)
filtered_spectrum = np.multiply(spectrograph,eqfilter)

# plt.plot(freq,eqfilter)
# plt.show()

pp_signal = ifft(filtered_spectrum)
pp_signal = np.array(pp_signal.real)

# plt.plot(pp_signal.real)
# plt.show()

pp_signal = np.array(pp_signal)
pp_signal *= 32767 / np.max(np.abs(pp_signal))
pp_signal = pp_signal.astype(np.int16)
play = sa.play_buffer(pp_signal,1,2,sample_rate)

play.wait_done()
play.stop()


# onesec = np.linspace(0,1,num=sample_rate)

# pcE = (np.sin(onesec*E2)+np.sin(onesec*B2)+np.sin(onesec*E3))*np.exp(-onesec)
# pcD = (np.sin(onesec*D3)+np.sin(onesec*A3)+np.sin(onesec*D4))*np.exp(-onesec)
# pcA = (np.sin(onesec*A2)+np.sin(onesec*E3)+np.sin(onesec*A3))*np.exp(-onesec)

# output = np.concatenate((np.zeros(sample_rate),pcE,np.zeros(sample_rate),pcD,np.zeros(sample_rate),pcA))


# output *= 32767 / np.max(np.abs(output))
# hardclip(output,10000)

# output = output.astype(np.int16)

# play = sa.play_buffer(output,1,2,sample_rate)

# play.wait_done()
# play.stop()

# plt.plot(output)
# plt.show()

# spectrograph = fft(output)
# freq = sample_rate*fftfreq(output.shape[-1])
# plt.plot(freq,np.abs(spectrograph))
# plt.xlim(0,20000)

# plt.show()



#############
# apply eq filter
##############

# eqfilter = np.exp(-np.abs(freq**2)/(400**2))
# filtered_spectrum = np.multiply(spectrograph,eqfilter)
# plt.plot(freq,eqfilter)
# plt.show()

# pp_signal = ifft(filtered_spectrum)
# pp_signal = np.array(pp_signal.real)
# plt.plot(pp_signal.real)
# plt.show()



# pp_signal = np.array(pp_signal)
# print(pp_signal)
#pp_signal *= 32767 / np.max(np.abs(pp_signal))
# pp_signal = pp_signal.astype(np.int16)
# play = sa.play_buffer(pp_signal,1,2,sample_rate)

# play.wait_done()
# play.stop()


