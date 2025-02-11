import numpy as np
import matplotlib.pyplot as plt

signalLengthSeconds =3
samplingFreqPerSecond = 1000
t_vals = np.linspace(0,signalLengthSeconds,samplingFreqPerSecond*signalLengthSeconds)


def signal(amplitude:float, shift:float,freq:float,t_vals:np.ndarray[float])->np.ndarray[float]:
    """Generates datapoints for sinus signal

    Args:
        amplitude (float): Amplitude for signal
        shift (float): shit up and down vertical axsis
        freq (float): freqenci of signal
        t_vals (np.NDArray[float]): numpy array of evenly spaced t values

    Returns:
        numpy array contaning x(t)=A*sin(2pi*f*t)+c
    """
    return amplitude*np.sin(2*np.pi*freq*t_vals)+shift
    #can 2 pi be pre calculated and is that faster? 

def exponetialPart(freq:float,t_vals:np.ndarray[float])->np.ndarray[float]:
    """Returns real and imaginary part of exp(-2*pi*i*f*t) or exp(-w*t)

    Args:
        freq (float): frequancy
        t_vals (np.NDArray[float]): t values

    Returns:
        np.NDArray[float]: 2 d numpy array index 0 real part, index 1 imaginary part
    """
    negativ_angular_frequency_w = -2*np.pi*freq
    realPart = np.cos(t_vals*negativ_angular_frequency_w)
    imaginaryPart = np.sin(t_vals*negativ_angular_frequency_w)
    return np.array([realPart,imaginaryPart])

def f_hat(x_of_t, t_vals,scanRangeFreqMin,scanRangeFreqMax):
    """f hat or fourier transformation takes a signal x(t) and converts it to the freq domain

    Args:
        x_of_t (_type_): signal
        t_vals (_type_): time value
        scanRangeFreqMin (_type_): minimum freq
        scanRangeFreqMax (_type_): maximum freq

    Returns:
        _type_: index[0][0] real part of transformation, index[0][1] imaginary part of transformation, index[1] frqensy values
    """
    real_part = np.zeros(len(t_vals))
    imaginary_part = np.zeros(len(t_vals))
    freqencyValues = np.linspace(scanRangeFreqMin, scanRangeFreqMax, len(t_vals))

    for i, freq in enumerate(freqencyValues):
        exp_parts = exponetialPart(freq, t_vals)  
        real_part[i] = np.sum( exp_parts[0] * x_of_t )
        imaginary_part[i] =np.sum( exp_parts[1] * x_of_t)
    
    return np.array([real_part, imaginary_part]), freqencyValues

        
def calculateMagnitude(forierValues):
    """Compines real and imaginary values from forirer transformation and adjust it to reflect amplitude of each signal

    Args:
        forierValues (_type_): 2d list with real and imagnery part at index 0 and 1 respecally

    Returns:
        _type_: 1d list with the combined values
    """
    return np.sqrt(forierValues[0]**2 + forierValues[1]**2)/len(forierValues[0])*2


def serchForFreq(fData,xdata,cutOffFromMax=0.5):
    """Serches for top points in data relative to the highest point in the data.

    Args:
        fData (_type_): freqency data
        xdata (_type_): amplitude data
        cutOffFromMax (float, optional): cutt of given in decimal value, higher value filters lower amplitude signals,lower values includes more signals but can include noise. Defaults to 0.5.

    Returns:
        _type_:List of freqcensies found
    """
    maxPoint = np.max(xdata)*cutOffFromMax
    topPoints = []
    for i in range(1, len(xdata) - 1):
        if (xdata[i]>xdata[i-1]) and (xdata[i]>xdata[i+1]) and (xdata[i]>maxPoint):
            topPoints.append(fData[i])
    
    return np.array(topPoints)



# signal1 = signal(1,0,2,t_vals)
# signal2 = signal(1,0,5,t_vals)
# combinedSignal = signal1+signal2

# plt.subplots(2,3)

# plt.subplot(2,3,1)
# plt.title("Signal 1")
# plt.plot(t_vals,signal1)
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")


# plt.subplot(2,3,2)
# plt.title("Signal 2")
# plt.plot(t_vals,signal2)
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")


# plt.subplot(2,3,3)
# plt.title("Kobinasjon")
# plt.plot(t_vals,combinedSignal)
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")



# forierValues,freqValues = f_hat(combinedSignal,t_vals,0,10)

# plt.subplot(2,3,4)
# plt.title("Forier reell del (abs)")
# plt.plot(freqValues,np.abs(forierValues[0])/len(forierValues[0]))
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")


# plt.subplot(2,3,5)
# plt.title("Forier imaginær del (abs)")
# plt.plot(freqValues,np.abs(forierValues[1])/len(forierValues[1]))
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")



# plt.subplot(2,3,6)
# plt.title("Forier samlet, justert for magnitude")
# plt.plot(freqValues,calculateMagnitude(forierValues))
# plt.grid()
# plt.axvline(color = "black")
# plt.axhline(color = "black")


signalFreq = np.array([2,4,8,50,100,500,1000,1500,2000])
signalFreq = np.array([2,4,8])
signals = np.array([signal(1,0,freq,t_vals) for freq in signalFreq])
combinedSignal = sum(signals)

# forierValues,freqValues = f_hat(combinedSignal,t_vals,0,4000)
forierValues,freqValues = f_hat(combinedSignal,t_vals,0,15)

fourierMagnitudeData = calculateMagnitude(forierValues)
foundFreqVals = serchForFreq(freqValues,fourierMagnitudeData)
# print(foundFreqVals)
print(f"Found {len(foundFreqVals)} frequnceis")
for freqFound in foundFreqVals:
    print(f"Found frequce: {freqFound}Hz")

plt.title("Forier samlet, justert for magnitude")
plt.plot(freqValues,fourierMagnitudeData)
plt.grid()
plt.axvline(color = "black")
plt.axhline(color = "black")




plt.show()