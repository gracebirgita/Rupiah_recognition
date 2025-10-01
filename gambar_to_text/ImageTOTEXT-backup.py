from ultralytics import YOLO
import csv
import os

def open_txt(path):
    with open(path, 'r') as file:
        txt = file.readlines()
    file.close()
    os.remove('image_to_text/predicted_text/labels/valid.txt')
    os.rmdir('image_to_text/predicted_text/labels')
    os.rmdir('image_to_text/predicted_text/')
    return txt

def get_digit(text):
    if text == '0':
        return 'seratus', '100'
    elif text == '1' or text == '4':
        return 'seribu', '1000'
    elif text == '2':
        return 'sepuluh ribu', '10000'
    elif text == '3':
        return 'seratus ribu', '100000'
    elif text == '5':
        return 'dua ratus', '200'
    elif text == '6':
        return 'dua ribu', '2000'
    elif text == '7':
        return 'dua puluh ribu', '20000'
    elif text == '8':
        return 'lima ratus', '500'
    elif text == '9':
        return 'lima ribu', '5000'
    elif text == '10':
        return 'lima puluh ribu', '50000'
    


def image_to_text():
    path="ITT\epochs200_nocoin.pt"
    model = YOLO(path)

    img_path="C:/Users/graze/Documents/SEM-4/SPEECH_RECOGNITION/FINPRO/Riset/valid.jpg"

    results = model.track(img_path, conf=0.4, save_txt = True, name='predicted_text', project="image_to_text")
    #yolo predict project=PROJECT name=NAME save

    print(results)
    content = open_txt('image_to_text/predicted_text/labels/valid.txt')
    # content = open_text('')

    pred = {}
    huruf_pred={}
    total =0
    for i in range(len(content)):
        result = content[i][0]

        huruf, angka = get_digit(result)

        if angka not in pred and huruf not in huruf_pred:
            pred[angka] = 1
            huruf_pred[huruf]=1
            total += int(angka)
        else:
            pred[angka] += 1
            huruf_pred[huruf] +=1
            total += int(angka)

    print(pred) # angka
    print(huruf_pred)

    print(total)

image_to_text()