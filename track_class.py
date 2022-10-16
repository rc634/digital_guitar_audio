import numpy as np
import wave
from numpy.fft import fft
from numpy.fft import fftfreq
from numpy.fft import ifft
import simpleaudio as sa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class eq_curves:
	def __init__(self):
		
		# this corresponds to bass=3.2, mids=8.5, treble=6.8 on a marshall https://www.guitarscience.net/tsc/marshall.htm#RIN=1300&R1=33k&RT=220k&RB=1M&RM=25k&RL=517k&C1=470p&C2=22n&C3=22n&RB_pot=LogB&RM_pot=Linear&RT_pot=Linear
		self.marshall_387_Hz = [0.,10.,20.,30.,60.,90.,140.,200.,300.,400.,500.,600.,800.,1000.,1300.,2000.,3000.,4000.,5000.,7000.,10000.,15000.,20000.,25000.,30000.,35000.,40000.]
		self.marshall_387_dB = [-30.,-19.,-13.,-11.,-7.7,-6.7,-6.4,-6.7,-7.3,-7.9,-8.2,-8.2,-7.8,-7.3,-6.4,-4.9,-3.9,-3.5,-3.2,-3.,-2.9,-2.8,-2.8,-2.8,-2.8,-2.8,-2.8]
		
		# celestion vintage 30 speaker cab, https://celestion.com/wp-content/uploads/2019/09/T3903-Vintage-30-copy.jpg
		self.CV30_Hz=[0.,10.,20.,30.,40.,50.,70.,100.,200.,300.,400.,500.,700.,1000.,2000.,3000.,4000.,5000.,6000.,7000.,8000.,9000.,10000.,15000.,20000.,30000.,40000.]
		self.CV30_dB=[50.,65.,76.,83.,85.,87.,92.,95.,98.,98.,98.,100.,101.,101.,103.,105.,103.,98.,87.,82.,77.,77.,80.,75.,75.,73.,70.]
		
		#fender bassman amp with bass=4.4, mids = 6.5, treble=7.4, https://www.guitarscience.net/tsc/bassman5f6a.htm#RIN=1300&R1=56k&RT=250k&RB=1M&RM=25k&RL=1M&C1=250p&C2=20n&C3=20n&RB_pot=LogA&RM_pot=Linear&RT_pot=Linear
		self.fender_467_Hz=[0.,10.,20.,30.,40.,50.,60.,80.,100.,200.,300.,400.,500.,600.,800.,1000.,2000.,3000.,4000.,5000.,7000.,10000.,20000.,30000.,40000.]
		self.fender_467_dB=[-18.,-12.2,-7.7,-5.9,-5.2,-4.8,-4.8,-4.9,-5.2,-7.7,-9.9,-11.4,-12.2,-12.5,-12.,-11.,-7.,-5.,-4.2,-3.6,-3.1,-2.8,-2.5,-2.5,-2.5]



class track_class:
	def __init__(self,filename,sample_rate):
		self.filename = filename
		self.sample_rate = sample_rate
		self.normalisation_factor = 0.85 # the loudest a signal will be with respect to digital headroom of integer
		self.audio = "N/A"
		self.spectrograph = "N/A" #  the complex version of the spectrum, includes both real spectrum and imaginary spectrum
		self.spectrum = "N/A" # real part of fourrier spectrum
		self.spectrum_im = "N/A" # imaginary part of fourrier spectrum
		self.fft_freq = "N/A" # the frequency array of the fourrier spectrum
		self.time = "N/A"
		self.raw_audio = "N/A"
		self.total_time = "N/A"
		self.load_audio_file_to_nparray()
		self.EQ = eq_curves()

	def normalise(self, ratio="N/A"):
		if ratio == "N/A":
			ratio = self.normalisation_factor
		self.audio = self.audio*32767 / np.max(np.abs(self.audio)) * ratio

	def load_audio_file_to_nparray(self):
		ifile = wave.open(self.filename)
		samples = ifile.getnframes()
		audio = ifile.readframes(samples)
		audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)[::2]
		audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
		max_int16 = 2**15
		audio_normalised = audio_as_np_float32 / max_int16
		raw_audio = audio_normalised*32767 / np.max(np.abs(audio_normalised)) * self.normalisation_factor
		self.audio = raw_audio
		self.raw_audio = raw_audio
		self.time = np.linspace(0.,float(len(raw_audio))/self.sample_rate,len(raw_audio))
		self.total_time = float(len(raw_audio))/self.sample_rate

	def play_audio(self):
		playable_audio = self.audio.astype(np.int16) # make into int so can play audio
		play = sa.play_buffer(playable_audio,1,2,self.sample_rate)
		play.wait_done()
		play.stop()

	def plot_waveform(self,start_time=0., end_time=-1.):
		plt.plot(self.time,self.audio/32767.)
		if end_time > start_time:
			plt.xlim(start_time,end_time)
		plt.show()

	def plot_waveform_vs_raw(self,start_time=0., end_time=-1.):
		plt.plot(self.time,self.raw_audio/32767.)
		plt.plot(self.time,self.audio/32767.)
		if end_time > start_time:
			plt.xlim(start_time,end_time)
		plt.show()

	def fft(self):
		self.spectrograph = fft(self.audio)
		self.spectrum = self.spectrograph.real
		self.spectrum_im = self.spectrograph.imag
		self.fft_freq = self.sample_rate*fftfreq(self.audio.shape[-1])

	def plot_spectrum(self,f_0=0.,f_1=-1.):
		if self.spectrum =="N/A":
			print("###########\nMust used self.fft before spectrum plot\n###########")
			return
		plt.plot(self.fft_freq,self.spectrum)
		if f_1 > f_0:
			plt.xlim(f_0,f_1)
		plt.show()

	def plot_power_spectrum(self,f_1=-1.):
		if self.spectrum =="N/A":
			print("###########\nMust used self.fft before spectrum plot\n###########")
			return
		plt.plot(self.fft_freq,self.spectrum**2 + self.spectrum_im**2)
		if f_1 > 0.:
			plt.xlim(0.,f_1)
		plt.show()

	def plot_log_power_spectrum(self):
		plt.loglog(self.fft_freq,self.spectrum**2 + self.spectrum_im**2)
		plt.show()

	def invfft(self):
		self.spectrum = self.spectrograph.real
		self.spectrum_im = self.spectrograph.imag
		self.fft_freq = self.sample_rate*fftfreq(self.audio.shape[-1])
		pp_signal = ifft(self.spectrograph)
		self.audio = np.array(pp_signal.real)

	# headroom means less distortion, generally headroom > 1 is pretty clean
	def sigmoid_clip(self,headroom,asymmetry = 0.02):
		headroom *= 32767.
		self.audio =  2./(1.+np.exp(-(5.*self.audio)/headroom)) -1.0 - asymmetry
		self.normalise()

	# enter hedroom h, 0<h<1 as a fraction of max headroom to clip
	def hardclip(self,headroom):
		headroom = headroom*32767
		ups = np.where(self.audio > headroom)
		downs = np.where(self.audio < -headroom)
		self.audio[ups[0]] = headroom
		self.audio[downs[0]] = -headroom
		self.normalise()

	def volume(self,volume):
		self.audio *= volume

	def blend_clean(self,ratio):
		self.audio += ratio*self.raw_audio
		self.normalise()

	# sharp cutoff below bassfreq, smooth cutoff above treblefreq
	def apply_cab_sim(self,bassfreq=100.,treblefreq=10000.,midboost=True):
		self.fft()
		eqfilter = np.exp(-np.abs(self.fft_freq**2)/(treblefreq**2))
		eqfilter -= np.exp(-np.abs(self.fft_freq**2)/bassfreq**2)
		if midboost:
			eqfilter += np.exp(-((np.abs(self.fft_freq)-2000.)**2)/(800.**2))
		self.spectrograph = np.multiply(self.spectrograph,eqfilter)
		self.invfft()
		self.normalise()

	def plot_cab_sim_eq_curve(self,bassfreq=100.,treblefreq=10000.,midboost=True): 
		f = np.linspace(0.,40000.,100000)
		eqfilter = np.exp(-(f**2)/(treblefreq**2))
		eqfilter -= np.exp(-(f**2)/(bassfreq**2))
		if midboost:
			eqfilter += np.exp(-((f-2000.)**2)/(800.**2))
		plt.loglog(f,eqfilter)
		plt.show()

	# bass middle treble filter
	# a = amplitude 0 to 1, f = frequency, sig = width
	#midbands is a list of bands
	def bmt_filter(self,ab,wb,sigb,midbands,at,wt,sigt):
		self.fft()
		f=self.fft_freq
		eqfilter = 1. + ab/(1.+ np.exp((f-wb)/sigb))
		eqfilter += at/(1.+ np.exp(-(f-wt)/sigt))
		for band in midbands :
			am = band[0]
			wm = band[1]
			sigm = band[2]
			eqfilter += am*np.exp(-((f-wm)/(sigm))**2)
		self.spectrograph = np.multiply(self.spectrograph,eqfilter)
		self.invfft()
		self.normalise()

	# bass middle treble filter
	# a = amplitude -1 to inf, f = frequency, sig = width
	#midbands is a list of bands
	def plot_bmt_filter(self,ab,wb,sigb,midbands,at,wt,sigt):
		f = np.linspace(0.,40000.,100000)
		eqfilter = 1. + ab/(1.+ np.exp((f-wb)/sigb))
		eqfilter += at/(1.+ np.exp(-(f-wt)/sigt))
		for band in midbands :
			am = band[0]
			wm = band[1]
			sigm = band[2]
			eqfilter += am*np.exp(-((f-wm)/(sigm))**2)
		plt.plot(f,20*np.log(eqfilter))
		plt.xlim(10,30000)
		plt.xscale("log")
		plt.show()

	def apply_fender_eq(self):
		self.bmt_filter(-0.8,5.,5.,[[-0.6,300.,200.,],[-0.5,643.,600.]],2.,3000.,1000.)

	def plot_fender_eq(self):
		self.plot_bmt_filter(-0.8,5.,5.,[[-0.6,300.,200.,],[-0.5,643.,600.]],2.,3000.,1000.)

	def apply_marshall_eq(self):
		self.bmt_filter(-10.0,10.,10.,[[3.5,150.,300.,],[1.2,1000.,400.],[1.2,2000.,800.],[1.2,4000.,1600.]],11.,1200.,4000.)

	def plot_marshall_eq(self):
		self.plot_bmt_filter(-10.0,10.,10.,[[3.5,150.,300.,],[1.2,1000.,400.],[1.2,2000.,800.],[1.2,4000.,1600.]],11.,1200.,4000.)


	# decent reprersentation fo CV30 cab (i checked in dB space)
	def plot_CV30_cab_sim(self):
		self.plot_bmt_filter(-1.0,50.,50.,[[0.45,3000.,1300.]],-0.7,5200.,2000.)

	# decent reprersentation fo CV30 cab (i checked in dB space)
	def apply_CV30_cab_sim(self):
		self.bmt_filter(-1.0,50.,50.,[[0.45,3000.,1300.]],-0.7,5200.,2000.)

	def plot_basscut(self):
		self.plot_bmt_filter(-1.0,40.,40.,[],0.0,5200.,2000.)

	def apply_basscut(self):
		self.bmt_filter(-1.0,40.,40.,[],0.0,5200.,2000.)

	#uses cubic interpolation to sample points given to it and return smooth eq curve
	#should start at 0Hz and end at like 40kHz
	def apply_cubic_eq(self,dB_array,Hz_array):	
		self.fft()
		fnew = np.linspace(0.,40000.,100000)

		#f = interp1d(x, y) # linear interp
		dB_interp = interp1d(np.array(Hz_array), np.array(dB_array), kind='cubic')
		eqfilter = np.exp(np.array(dB_interp(fnew))/20.)

		self.spectrograph = np.multiply( np.interp(self.fft_freq,fnew,eqfilter) , self.spectrograph ) 
		self.invfft()
		self.normalise()

	#uses cubic interpolation to sample points given to it and return smooth eq curve
	#should start at 0Hz and end at like 40kHz
	def plot_cubic_eq(self,dB_array,Hz_array):	
		self.fft()
		fnew = np.linspace(0.,40000.,100000)

		#f = interp1d(x, y) # linear interp
		dB_interp = interp1d(np.array(Hz_array), np.array(dB_array), kind='cubic')
		eqfilter = np.exp(np.array(dB_interp(fnew))/20.)

		self.spectrograph = np.multiply( np.interp(self.fft_freq,fnew,eqfilter) , self.spectrograph ) 

		plt.plot(Hz_array,dB_array,'k')

		#plt.plot(self.fft_freq,20.*np.log(np.interp(self.fft_freq,fnew,eqfilter)),'r')

		plt.plot(fnew,20*np.log(eqfilter),'r')

		plt.legend(['Input data points','Interpolated function'])

		plt.xscale("log")
		plt.xlim(10,30000)
		plt.show()




