import sys
import csv
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import datetime
from pickle import dump

def read_file():
    all_data = []
    with open('./data/apple_2010_2018_truth.csv', encoding="utf-8_sig") as f:
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

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size=0.9)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test

def build_model(steps):
    inputs = tf.keras.Input(shape=(steps,5))
    x = tf.keras.layers.LSTM(64, activation=tf.keras.activations.tanh)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.keras.activations.linear)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    all_data = read_file()
    mmscaler = sklearn.preprocessing.MinMaxScaler()
    mmscaler.fit(all_data)
    all_data = mmscaler.transform(all_data)
    dump(mmscaler, open("./model/apple_mmscaler.pkl", "wb"))

    steps=10
    x_train, x_test, y_train, y_test = make_datasets(all_data, steps)

    model = build_model(steps)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.summary()

    tfboard_cbks = tf.keras.callbacks.TensorBoard(log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
    modelsave_cbks = tf.keras.callbacks.ModelCheckpoint('./model/apple_ckpt_epoch_{epoch:04d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    cbks = [tfboard_cbks, modelsave_cbks]

    history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=500, verbose=2, callbacks=cbks, validation_data=(x_test, y_test), shuffle=True, validation_freq=1)

    score = model.evaluate(x=x_test, y=y_test, verbose=2)
    print('Test loss:', score[0])
    print('Test metrix:', score[1])

    pred = model.predict(x=x_test, batch_size=None, verbose=2, steps=None)
    pred = mmscaler.inverse_transform(pred)
    print(pred)

main()
