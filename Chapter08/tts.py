import pyttsx3;
engine = pyttsx3.init();
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-50)
voices = engine.getProperty('voices')
engine.setProperty('voice', 'TTS_MS_EN-US_ZIRA_11.0')
engine.say("You are reading the Python Machine Learning Cookbook");
engine.say("I hope you like it.");
engine.runAndWait();


