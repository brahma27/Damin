
import numpy as np
import pandas as pd
import collections
import csv

def data_latih():
    temp =[]
    file = open('data_latih.csv')
    d_latih = csv.reader(file)
    for baris in d_latih:
        temp.append(baris)
    return temp

def data_uji():
    tamp =[]
    uji = open('data_uji.csv')
    d_uji = csv.reader(uji)
    for baris in d_uji:
        tamp.append(baris)
    return tamp

def pisah_kelas(data):
    yes = []
    no = []
    for baris in data:
        if(baris[4]=='yes'):
            yes.append(baris)
        else:
            no.append(baris)
    return yes,no

def probabilitas_atribut(data):
    tampung = []
    rotasi = np.transpose(data)
    for i in rotasi:
        jum = collections.Counter(i) 
        tam = []
        for j in jum:
            tam.append([j,jum[j]/len(data)])
        tampung.append(tam)
    return tampung

def probabilitas_kelas(d1,d2):
    kelas = []
    ya,tidak= d1,d2
    panjang_latih = data_latih()
    P_yes = (len(ya)/len(panjang_latih))
    P_no =  (len(tidak)/len(panjang_latih))
    kelas.append(P_yes)
    kelas.append(P_no)
    return kelas

def cek(uji,data):
    for baris in data:
        if(baris[0]==uji):
            return baris[1]

############============MAIN==================##########################
if __name__ == '__main__':
    hasil = []
    i =1
    latih = data_latih()
    uji = data_uji()
    ya,tidak = pisah_kelas(latih)
    prob_kelas = probabilitas_kelas(ya,tidak)
    prob_yes = probabilitas_atribut(ya)
    prob_no =  probabilitas_atribut(tidak)
    for baris in uji:
        hasil_yes = prob_kelas[0]*(cek(baris[0],prob_yes[0])*cek(baris[1],prob_yes[1])*cek(baris[2],prob_yes[2])*cek(baris[3],prob_yes[3]))
        hasil_no = prob_kelas[1]*(cek(baris[0],prob_no[0])*cek(baris[1],prob_no[1])*cek(baris[2],prob_no[2])*cek(baris[3],prob_no[3]))
        print(hasil_yes)
        print(hasil_no)
        if(hasil_yes>hasil_no):
            print('Keputusan ke',i,': yes')
            hasil.append([baris[0],baris[1],baris[2],baris[3],'yes'])
        else:
            print('Keputusan ke',i,': no')
            hasil.append([baris[0],baris[1],baris[2],baris[3],'no'])
        i+=1
        break
    pilihan = open('prediksi.csv', 'w')
    jaw = csv.writer(pilihan, lineterminator='\n')
    for data in hasil:
            jaw.writerow(data)
                
    from sklearn.metrics import confusion_matrix

    file = open('data_target.csv')
    d_latih = csv.reader(file)
    simpan = []
    for baris in d_latih:
        simpan.append(baris)

    tampung = []
    for baris in hasil:
        tampung.append(baris[4:])

    target = simpan
    prediksi = tampung

    tn, fp, fn, tp = confusion_matrix(target, prediksi).ravel() 
    tn = (tn)
    fp = (fp)
    fn = (fn)
    tp = (tp)
    akurasi = ((tp+tn)/(tp+tn+fp+fn))*100
    presisi = (tp/(fp+tp))*100
    recall = (tp/(fn+tp))*100
    f1_score = 2* ((recall*presisi)/(recall+presisi))
    print('='*30)
    print ('akurasi ', akurasi)
    print ('presisi ',presisi)
    print ('recall ', recall)
    print('F1 Score ',f1_score)
