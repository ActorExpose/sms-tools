# function for doing a morph between two sounds using the hpsModel

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

def analysis(inputFile1='../../sounds/violin-B3.wav', window1='blackman', M1=1001, N1=1024, t1=-100, 
	minSineDur1=0.05, nH=60, minf01=200, maxf01=300, f0et1=10, harmDevSlope1=0.01, stocf=0.1,
	inputFile2='../../sounds/soprano-E4.wav', window2='blackman', M2=901, N2=1024, t2=-100, 
	minSineDur2=0.05, minf02=250, maxf02=500, f0et2=10, harmDevSlope2=0.01):
	"""
	Analyze two sounds with the harmonic plus stochastic model
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	M: analysis window size 
	N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks 
	minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics
	minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound
	f0et: maximum error accepted in f0 detection algorithm                                                                                            
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
	stocf: decimation factor used for the stochastic approximation
	returns inputFile: input file name; fs: sampling rate of input file,
	        hfreq, hmag: harmonic frequencies, magnitude; stocEnv: stochastic residual
	"""

	# size of fft used in synthesis
	Ns = 512
	# hop size (has to be 1/4 of Ns)
	H = 128
	# read input sounds
	(fs1, x1) = UF.wavread(inputFile1)
	(fs2, x2) = UF.wavread(inputFile2)
	# compute analysis windows
	w1 = get_window(window1, M1)
	w2 = get_window(window2, M2)
	# compute the harmonic plus stochastic models
	hfreq1, hmag1, hphase1, stocEnv1 = HPS.hpsModelAnal(x1, fs1, w1, N1, H, t1, nH, minf01, maxf01, f0et1, harmDevSlope1, minSineDur1, Ns, stocf)
	hfreq2, hmag2, hphase2, stocEnv2 = HPS.hpsModelAnal(x2, fs2, w2, N2, H, t2, nH, minf02, maxf02, f0et2, harmDevSlope2, minSineDur2, Ns, stocf)
	return inputFile1, fs1, hfreq1, hmag1, stocEnv1, inputFile2, hfreq2, hmag2, stocEnv2

def transformation_synthesis(inputFile1, fs, hfreq1, hmag1, stocEnv1, inputFile2, hfreq2, hmag2, stocEnv2,
	hfreqIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1]), hmagIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1]), stocIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1])):
	"""
	Transform the analysis values returned by the analysis function and synthesize the sound
	inputFile1: name of input file 1
	fs: sampling rate of input file	1
	hfreq1, hmag1, stocEnv1: hps representation of sound 1
	inputFile2: name of input file 2
	hfreq2, hmag2, stocEnv2: hps representation of sound 2
	hfreqIntp: interpolation factor between the harmonic frequencies of the two sounds, 0 is sound 1 and 1 is sound 2 (time,value pairs)
	hmagIntp: interpolation factor between the harmonic magnitudes of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
	stocIntp: interpolation factor between the stochastic representation of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
	"""
	
	# size of fft used in synthesis
	Ns = 512
	# hop size (has to be 1/4 of Ns)
	H = 128

	# morph the two sounds
	yhfreq, yhmag, ystocEnv = HPST.hpsMorph(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp)

	# synthesis 
	y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

	# write output sound 
	outputFile = 'output_sounds/' + os.path.basename(inputFile1)[:-4] + '_hpsMorph.wav'
	UF.wavwrite(y, fs, outputFile)

if __name__ == "__main__":
	# analysis
	inputFile1, fs1, hfreq1, hmag1, stocEnv1, inputFile2, hfreq2, hmag2, stocEnv2 = analysis()

	# transformation and synthesis
	transformation_synthesis (inputFile1, fs1, hfreq1, hmag1, stocEnv1, inputFile2, hfreq2, hmag2, stocEnv2)

	plt.show()
