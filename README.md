# LSTM-TimeSeries-Predictor
LSTM을 활용한 시계열 예측 문제

```py
data = np.sin(time_steps) + 0.1 * np.random.normal(size=len(time_steps))
```

```py
for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length]) 
```



#### 하이퍼파라미터 설정
```
sequence_length = 50
batch_size = 32
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 20
```
#### Sin 예측 파형
![Plot](./outputs/plot.png)

#### 