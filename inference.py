import sys
import csv
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import datetime
from pickle import load

def read_file():
    all_data = []
    with open('./data/GM_truth_2018to2020.csv', encoding="utf-8_sig") as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            day_data = []
            #Date, Open, Close, High, Low, Volume
            # day_data.append(str(line[0].replace(' ', '')))
            day_data.append(float(line[3].replace(' ', '').replace('$', '')))
            day_data.append(float(line[1].replace(' ', '').replace('$', '')))
            day_data.append(float(line[4].replace(' ', '').replace('$', '')))
            day_data.append(float(line[5].replace(' ', '').replace('$', '')))
            day_data.append(int(line[2].replace(' ', '')))
            all_data.append(day_data)

        all_data = all_data[::-1]
        return(all_data)

def make_datasets(all_data, steps):
    x_all = []
    y_all= []
    for index in range(len(all_data)):
        if index + steps > len(all_data) - 1:
            break
        x_all.append(all_data[index:index+steps])
        y_all.append(all_data[index+steps])

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    return x_all, y_all

def read_model():
    model = tf.keras.models.load_model('./model/GM_ckpt_epoch_0111.hdf5')
    return model

def main():
    all_data = read_file()
    mmscaler = load(open("./model/GM_mmscaler.pkl", "rb"))
    mmscaler.fit(all_data)
    all_data = mmscaler.transform(all_data)

    steps=10
    x_all, y_all = make_datasets(all_data, steps)

    model = read_model()
    model.summary()

    pred = model.predict(x=x_all, batch_size=None, verbose=2, steps=None)

    pred = mmscaler.inverse_transform(pred)
    with open("./data/GM_truth_2018to2020_predict.csv", "w", encoding="utf-8_sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(pred)

    all_data = mmscaler.inverse_transform(all_data)
    with open("./data/GM_truth_2018to2020_reversed.csv", "w", encoding="utf-8_sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(all_data)

main()
