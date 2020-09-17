import random 
import torch
import json
from model import NeuralNet
from Divyansh2 import Tokenize,bag_of_words
import pyttsx3
import speech_recognition as sr

with open("intents.json","r") as f:
    x = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size) 
model.load_state_dict(model_state)   
model.eval()

def speak(text):
    x = pyttsx3.init()
    x.say(text)
    x.runAndWait()
    
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))
            
    return said
    
bot_name = "Jarvis" 
speak("I m Jarvis type Mr Divyansh")
while True:
    sentence = get_audio()
    if sentence == "quit":
        speak("Bye Mr Divyansh")
        break
            
    sentence = Tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X =torch.from_numpy(X)
    
    output = model(X)
    _,pred = torch.max(output,dim=1)
    tag = tags[pred.item()]
    
    probs = torch.softmax(output,dim=1)
    prob = probs[0][pred.item()]
    
    if prob.item() > 0.75:
        for intent in x["intents"]:
            if tag == intent["tag"]:
                speak(random.choice(intent['responses']))
    
    else:  
        speak("Sorry don't know about this")
       
    
    
    
    
    
