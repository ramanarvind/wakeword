import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_distances
from fastdtw import dtw, fastdtw
import matplotlib.pyplot as plt

def preemphasis(signal, alpha=0.97):
    emphasizedSignal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasizedSignal

def getFileProp(file):
    signal, samplerate = sf.read(file)
    lenInMsec = len(signal) * 1000 / samplerate
    return lenInMsec

def processFile(file, window=0, offset=0):
    signal, samplerate = sf.read(file)

    if (offset + window) * samplerate / 1000 > len(signal):
        return False, 0
    
    if offset > 0 or window > 0:
        offsetInsamples = int(offset * samplerate / 1000)
        windowInSamples = int(window * samplerate / 1000)
        signal = signal[offsetInsamples : offsetInsamples + windowInSamples]

    si = 1 / samplerate
    t = np.arange(0, len(signal)*si, si)

    emphasizedSignal = preemphasis(signal)

    # 20msec frame size and 10msec stride
    frmSize = 20/1000
    frmStride = 10/1000
    samplesInSignal = len(signal)
    samplesInFrm = int(round(frmSize * samplerate, 0))
    strideInSamples = int(round(frmStride * samplerate, 0))
    numFrms = int((samplesInSignal - samplesInFrm) / strideInSamples)

    padLen = numFrms * strideInSamples + samplesInFrm - samplesInSignal
    if padLen > 0:
        z = np.zeros(padLen)
        emphasizedSignal = np.append(emphasizedSignal, z)

    idx = np.tile(np.arange(0, samplesInFrm), (numFrms, 1)) + np.tile(np.arange(0, numFrms * strideInSamples, strideInSamples), (samplesInFrm, 1)).T
    frms = emphasizedSignal[idx]

    # Apply a hamming window
    frms *= np.hamming(samplesInFrm)
    nFFT = 512
    freqDist = np.fft.rfft(frms, nFFT)
    energyFreqDist = 1.0/nFFT * (freqDist ** 2)

    # Create filter banks
    nFilt = 40
    lowFreqMel = 0
    highFreqMel = 2595 * np.log10(1 + (samplerate/2) / 700)
    melPoints = np.linspace(lowFreqMel, highFreqMel, nFilt + 2)
    melPointsInHz = (700 * (10**(melPoints/2595) - 1))
    bins = np.floor((nFFT + 1) * melPointsInHz / samplerate)

    filterBank = np.zeros((nFilt, int(np.floor(nFFT/2 + 1))))
    for m in range(1, nFilt + 1):
        binMinus = int(bins[m-1])
        bin = int(bins[m])
        binPlus = int(bins[m+1])
        for k in range(binMinus, bin):
            filterBank[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
        for k in range(bin, binPlus):
            filterBank[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])

    filterBankOutput = np.dot(energyFreqDist, filterBank.T)
    filterBankOutput = np.where(filterBankOutput == 0, np.finfo(float).eps, filterBankOutput)
    filterBankOutput = 20 * np.log10(filterBankOutput)

    numCeps = 12
    mfcc = dct(filterBankOutput, type=2, axis=1, norm='ortho')
    mfcc = mfcc[:, 1:numCeps+1]
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return True, mfcc.astype(np.float32)

def computeDis(x, y):
    cosineDis = cosine_distances([x], [y])
    return cosineDis[0]

refFile = './dataset/arvind1.wav'
lenRefSignal = lenTestSignal = getFileProp(refFile)
_, mfcc0 = processFile(refFile)

testFile = './dataset/arvind2.wav'
lenTestSignal = getFileProp(testFile)
for offset in range(0, int(lenTestSignal - lenRefSignal), 20):
    _, mfcc1 = processFile(testFile, lenRefSignal, offset)
    distance, path = fastdtw(mfcc0, mfcc1, radius=10, dist=computeDis)
    distance /= (len(mfcc0) + len(mfcc1))
    if distance[0] < 0.30:
        print("Wake word detected at offset=", offset, "msec. Distance = ", round(distance[0], 2))
    else:
        print("Processing file at offset=", offset, "msec. Distance = ", round(distance[0], 2))