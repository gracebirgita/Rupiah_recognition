from ultralytics import YOLO
import os

def open_txt(path):
    with open(path, 'r') as file:
        txt = file.readlines()
    file.close()
    os.remove('gambar_to_text/image_to_text/predicted_text/labels/place_holder.txt')
    os.rmdir('gambar_to_text/image_to_text/predicted_text/labels')
    os.rmdir('gambar_to_text/image_to_text/predicted_text/')
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
    path="epochs200_nocoin.pt"
    model = YOLO(path)

    img_path="image/place_holder.jpg"

    results = model.track(img_path, conf=0.4, save_txt = True, name='predicted_text', project="gambar_to_text/image_to_text")
    #yolo predict project=PROJECT name=NAME save

    # print(results)
    content = open_txt('gambar_to_text/image_to_text/predicted_text/labels/place_holder.txt')
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



    return pred, huruf_pred, total

'''
pred: output dengan format pred[digit] = total_angka_nya
* pred[100] = 2
* nominal 100 ada 2

huruf: output dengan format huruf[angka] = total_angka_nya
* huruf['dua ribu'] = 3
* angka 'dua ribu' ada 3

total: output total dari seluruh angka hasil pred
* {'10000' : 1, '20000' : 2}
* total = 50000 (integer)

* Cara pakai:
pred, huruf, total = image_to_text()

'''
pred, huruf, total = image_to_text()


# def nominal(angka):
#     if angka == '1':
#         return str('satu')
#     elif angka == '2':
#         return str('dua')
#     elif angka == '3':
#         return str('tiga')
#     elif angka == '4':
#         return str('empat')
#     elif angka == '5':
#         return str('lima')
#     elif angka == '6':
#         return str('enam')
#     elif angka == '7':
#         return str('tujuh')
#     elif angka == '8':
#         return str('delapan')
#     elif angka == '9':
#         return str('sembilan')
#     elif angka == '0':
#         return str('nol')
    

# def total_to_text(total):
#     total = str(total)
#     rev_total = total[::-1]
#     nominal_text = ''
#     for i in range(len(rev_total)):
#         if rev_total[i] == '0':
#             continue
#         if i == 0:
#             nominal_text = nominal(rev_total[i]) + ' ' + nominal_text
#         elif i == 1:
#             if rev_total[i] == '1':
#                 if rev_total[i-1] == '0':
#                     nominal_text = 'sepuluh ' + nominal_text
#                 elif rev_total[i-1] == '1':
#                     nominal_text = 'sebelas ' + nominal_text
#                 else:
#                     nominal_text = nominal(rev_total[i-1]) + ' belas ' + nominal_text
#             else:
#                 nominal_text = nominal(rev_total[i]) + ' puluh ' + nominal_text
#         elif i == 2:
#             if rev_total[i] == '1':
#                 nominal_text = 'seratus ' + nominal_text
#             else:
#                 nominal_text = nominal(rev_total[i]) + ' ratus ' + nominal_text
#         elif i == 3:
#             nominal_text = nominal(rev_total[i]) + ' ribu ' + nominal_text
#         elif i == 4:
#             nominal_text = nominal(rev_total[i]) + ' puluh ' + nominal_text
#         elif i == 5:
#             nominal_text = nominal(rev_total[i]) + ' ratus ' + nominal_text
#         elif i == 6:
#             nominal_text = nominal(rev_total[i]) + ' juta ' + nominal_text

#     return nominal_text.strip()


# a = 'satu'
# b = 'dua'
# c = a + ' ' +b
# print(c)
# pred, huruf, total = image_to_text()
# nominal = total_to_text(121112)
# print(nominal)
