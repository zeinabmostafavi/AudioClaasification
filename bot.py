import telebot
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile

model.load_state_dict(torch.load("weights.pth"))
model.eval()

ID = ['sajjad', 'zeynab', 'amir', 'hossein', 'parisa', 'nahid', 'maryam', 'alireza', 'zahra', 'morteza', 'mohammadali']


bot = telebot.TeleBot("")
@bot.message_handler(commands=['start'])
def say_hi(messages):
    bot.send_message(
        messages.chat.id, f'Hi {messages.from_user.first_name} Dear😎 ')
    bot.send_message(
        messages.chat.id, f' Now send me your voice  ☺...')
@bot.message_handler(content_types=['voice'])
def voice(message):
    audio_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(audio_info.file_path)
    src = audio_info.file_path

    with open(src, 'wb') as audio_file:
        audio_file.write(downloaded_file)
    
    signal, sample_rate = torchaudio.load(src)
    
    signal = torch.mean(signal, dim=0, keepdim=True)
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    signal = transform(signal)
    signal = signal.unsqueeze(0).to(device)
     
    preds = model(signal)
   
    preds = preds.cpu().detach().numpy()
    output = np.argmax(preds)
    
    bot.reply_to(message,"You Are "+ID[output]+" :D")
    print(output)

bot.polling()