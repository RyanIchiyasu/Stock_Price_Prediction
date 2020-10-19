# Stock_Price_Prediction
■ Data
Train:General Motors2010-2017(open, close, high, low, volume)/day
Test:General Motors2018-2020(open, close, high, low, volume)/day

■ I/O
Input:(open, close, high, low, volume) 10days(n-10 to n-1)
Output:(open, close, high, low, volume)  1day(n)

■ Model archtecture
・1 LSTM layer 64 cells
・Many to one (10, 5) ⇒ (5, )
・optimizer:adam, loss:MSE"								

<img src="https://github.com/RyunosukeIchiyasu/Stock_Price_Prediction/blob/main/data/GM_sample.PNG" width="1000">
