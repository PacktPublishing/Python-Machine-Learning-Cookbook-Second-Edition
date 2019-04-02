import numpy as np
import time
from matplotlib import pyplot

np.random.seed(1)
states = ["Sunny","Rainy"]

TransStates = [["SuSu","SuRa"],["RaRa","RaSu"]]
TransnMatrix = [[0.75,0.25],[0.30,0.70]]

if sum(TransnMatrix[0])+sum(TransnMatrix[1]) != 2:
    print("Warning! Probabilities MUST ADD TO 1. Wrong transition matrix!!")
    raise ValueError("Probabilities MUST ADD TO 1")

WT = list()
NumberDays = 200
WeatherToday = states[0]

print("Weather initial condition =",WeatherToday)

i = 0

while i < NumberDays:
    
    if WeatherToday == "Sunny":        
        TransWeather = np.random.choice(TransStates[0],replace=True,p=TransnMatrix[0])
        if TransWeather == "SuSu":
            pass
        else:
            WeatherToday = "Rainy"


            
    elif WeatherToday == "Rainy":
        TransWeather = np.random.choice(TransStates[1],replace=True,p=TransnMatrix[1])
        if TransWeather == "RaRa":
            pass
        else:
            WeatherToday = "Sunny"

            
    print(WeatherToday)
    WT.append(WeatherToday) 
    i += 1
    time.sleep(0.2)

pyplot.plot(WT)
pyplot.show()


