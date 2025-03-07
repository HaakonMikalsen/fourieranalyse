import numpy as np
import matplotlib.pyplot as plt


def signal(amplitude:float, shift:float,freq:float,tData:np.ndarray[float],offset = 0)->np.ndarray[float]:
    """Generates datapoints for sinus signal

    Args:
        amplitude (float): Amplitude for signal
        shift (float): shit up and down vertical axsis
        freq (float): freqenci of signal
        tData (np.NDArray[float]): numpy array of evenly spaced t values
        offset () : offsets signal along the horizantal axsis

    Returns:
        numpy array contaning x(t)=A*sin(2pi*f*t)+c
    """
    return amplitude*np.sin(2*np.pi*freq*tData +offset)+shift
    

def exponetialPartFHat(freq:float,tData:np.ndarray[float])->np.ndarray[float]:
    """Returns real and imaginary part of exp(-2*pi*i*f*t) or exp(-w*t)

    Args:
        freq (float): frequancy
        tData (np.NDArray[float]): t values

    Returns:
        np.NDArray[float]: 2 d numpy array index 0 real part, index 1 imaginary part
    """
    negativ_angular_frequency_w = -2*np.pi*freq
    realPart = np.cos(tData*negativ_angular_frequency_w)
    imaginaryPart = np.sin(tData*negativ_angular_frequency_w)
    return np.array([realPart,imaginaryPart])

def f_hat(x_of_t, tData,scanRangeFreqMin,scanRangeFreqMax):
    """f hat or fourier transformation takes a signal x(t) and converts it to the freq domain

    Args:
        x_of_t (_type_): signal
        tData (_type_): time value
        scanRangeFreqMin (_type_): minimum freq
        scanRangeFreqMax (_type_): maximum freq

    Returns:
        _type_: index[0][0] real part of transformation, index[0][1] imaginary part of transformation, index[1] frqensy values
    """
    real_part = np.zeros(len(tData))
    imaginary_part = np.zeros(len(tData))
    freqencyValues = np.linspace(scanRangeFreqMin, scanRangeFreqMax, len(tData))

    N = len(tData)

    for i, freq in enumerate(freqencyValues):
        exp_parts = exponetialPartFHat(freq, tData)  
        real_part[i] = np.sum( exp_parts[0] * x_of_t )/N
        imaginary_part[i] =np.sum( exp_parts[1] * x_of_t)/N
    
    return np.array([real_part, imaginary_part]), freqencyValues

        
def calculateMagnitude(forierValues):
    """Compines real and imaginary values from forirer transformation and adjust it to reflect amplitude of each signal

    Args:
        forierValues (_type_): 2d list with real and imagnery part at index 0 and 1 respecally

    Returns:
        _type_: 1d list with the combined values
    """
    return np.sqrt(forierValues[0]**2 + forierValues[1]**2)


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
    topPointsFreq = []
    topPointsAmplitude = []
    for i in range(1, len(xdata) - 1):
        if (xdata[i]>xdata[i-1]) and (xdata[i]>xdata[i+1]) and (xdata[i]>maxPoint):
            topPointsFreq.append(fData[i])
            topPointsAmplitude.append(xdata[i])
    
    return np.array([topPointsFreq,topPointsAmplitude])


def createSignalFromFreq(freqencis,tData,ampltudes=[],offsetX=[],offsetY=[]):
    if len(ampltudes)==0:
        ampltudes = np.linspace(1,1,len(freqencis))
    
    if len(offsetX)==0:
        offsetX = np.zeros(len(freqencis))
    
    if len(offsetY)==0:
        offsetY = np.zeros(len(freqencis))
    
    xData = np.zeros(len(tData))
    for i in range(len(freqencis)):
        xData+=signal(ampltudes[i],offsetY[i],freqencis[i],tData,offsetX[i])
    return xData


def exponetialPartF(freq:float,tData:np.ndarray[float])->np.ndarray[float]:
    negativ_angular_frequency_w = 2*np.pi*freq
    realPart = np.cos(tData*negativ_angular_frequency_w)
    imaginaryPart = np.sin(tData*negativ_angular_frequency_w)
    return np.array([realPart,imaginaryPart])

def f(f_hat_values,freqencyValues,tData):
    f_values_real = np.zeros(len(tData))
    f_values_imaginary = np.zeros(len(tData))

    N = len(tData)

      
    for i, freq in enumerate(freqencyValues):
        exp_parts = exponetialPartF(freq, tData)  
        f_values_real += exp_parts[0] * f_hat_values[0][i]
        f_values_imaginary+= exp_parts[1] * f_hat_values[1][i]

    
    
    return f_values_real,f_values_imaginary








timeStepInSeconds =20e-3

filePath = r"./heartData1.txt"

file = open(filePath)

tData =[]
xData =[]

for i,line in enumerate(file):
    tData.append(i*timeStepInSeconds)
    xData.append(int(line))

plotWidthInSeconds = 3
plotWidthInSteps = int(plotWidthInSeconds/timeStepInSeconds)

tData = np.array(tData)
xData = np.array(xData)





forierValues,freqValues = f_hat(xData,tData,5,35)

fourierMagnitudeData = calculateMagnitude(forierValues)
foundFreqVals = serchForFreq(freqValues,fourierMagnitudeData)

print(f"Found {len(foundFreqVals[0])} frequnceis")
for i in range(len(foundFreqVals[0])):
    print(f"Found frequce: {foundFreqVals[0][i]}Hz with amplitude {foundFreqVals[1][i]}")
plt.title("Forier samlet, justert for magnitude")
plt.plot(freqValues,fourierMagnitudeData)

plt.grid()
plt.axvline(color = "black")
plt.axhline(color = "black")

plt.show()



reversed_transfor = f(forierValues,freqValues,tData)
adjust = np.average(np.abs(xData))/np.average(np.abs(reversed_transfor[0])) 

plt.subplot(1,3,1)
plt.plot(tData,reversed_transfor[0]*adjust)
plt.subplot(1,3,2)
plt.plot(tData,xData,color="red")
plt.subplot(1,3,3)
plt.plot(tData,xData,color="red")
plt.plot(tData,reversed_transfor[0]*adjust)
plt.show()


xDataNormalized = (reversed_transfor[0]-np.median(reversed_transfor[0]))/(np.percentile(reversed_transfor[0],75)-np.percentile(reversed_transfor[0],25))


for i in range(len(reversed_transfor[0])):
    plt.clf()
    plt.plot(tData[i:(i+plotWidthInSteps)], xDataNormalized[i:(i+plotWidthInSteps)]*adjust,color="red")
    plt.pause(timeStepInSeconds)

plt.show()