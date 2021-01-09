# 4. Sequences, Time Series and Prediction

## WEEK 1 : Sequences and Prediction

## 학습 내용
시퀀스 모델, 시계열 데이터에 대해 배우고, 먼저 이러한 기술을 연습하고 인공적인 데이터를 기반으로 이러한 모델을 구축하는 것<br>

### Time Series
시계열은 어디에나 존재<br>
주식 가격, 일기 예보, 무어의 법칙과 같은 역사적 동향에도 가능<br>
시간 요소가있는 모든 것은이 방법으로 분석 될 수 있음<br>
시계열은 모든 모양과 크기로 제공되지만 매우 일반적인 패턴이 많이 있음<br>
- 일변량 시계열 (Univariate time series)
  - 시간 당 온도
- 다변량 시계열
  - 시간 당 날씨<br>
  - 다변량 시계열 차트는 관련 데이터의 영향을 이해하는 유용한 방법<br>

[Week 1 실습_time series](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_1_Lesson_2.ipynb)<br>
데이터를 기반으로 예측<br>

#### 간단한 시계열(직선)
```
time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

#### imputation
이미 가지고있는 데이터보다 먼저 데이터를 수집 할 수 있었다면 데이터가 어떻게 될지 추측<br>
이미 존재하지 않는 데이터에 대해 데이터의 구멍을 채우기<br>

#### anomalies
시계열 예측을 사용하여 이상(anomalies)을 탐지<br>

#### Seasonality
계절 패턴 정의<br>
```
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
```

계절 패턴 추가<br>
```
baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

#### Trend + Seasonality
시계열이 움직이는 특정 방향을 갖는 추세<br>
일부 시계열에는 추세(Trend)와 계절성(Seasonality)이 모두 결합될 수 있음<br>
예측할 수 없는 것도 있음<br>
<br>

추세 추가
```
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```
계절 데이터가 시간이 지남에 따라 증가하는 형태

#### noise
노이즈 정의<br>
```
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
```
노이즈 추가<br>
```
noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()
```

#### trend + seansonality + noise
```
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

#### autoocrrelated time series (자동 상관 시계열)
각 시간 단계의 값이 이전 시간 단계 값의 99% 와 가끔 스파이크를 더한 것<br>
Data that follows a predictable shape, even if the scale is different<br>
<br>

autocorrelation 정의<br>
```
def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude
```
autocorrelation 적용<br>
```
series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()
```
- series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
추세, 계절성 추가하는 방법

#### Non-Stationary Time Series (비고정식 시계열)
상승세이다 어느 시점(Big event)부터 하락하는 모양<br>
과거 값을 기반으로 예측할 수 없음<br>
Big event 발생 후 하락세<br>
```
series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
series[200:] = series2[200:]
#series += noise(time, 30)
plot_series(time[:300], series[:300])
plt.show()
```

#### impulse

```
def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series 
```
```
series = impulses(time, 10, seed=42)
plot_series(time, series)
plt.show()
```

### Machine Learning을 사용하여 데이터를 이해하고 예측하는 방법을 배우고 싶다면 이와 같은 합성 데이터가 매우 유용하다는 것을 알아야함
<br>

[Week 1 실습_Forecasting](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%201%20-%20Lesson%203%20-%20Notebook.ipynb)<br>
<br>

### 예측 방법 : Naive forecasting 
마지막 값을 가져 와서 다음 값이 동일한 값이 될 것이라고 가정<br>
<br>

단지 기간 플러스 1의 값을 예측하는 것<br>
```
naive_forecast = series[split_time - 1:-1]
```
기간 지정해 확대해서 보기<br>
```
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)
```

### 모델 성능 측정
#### fixed partitioning
예측 모델의 성능을 측정하기 위해 일반적으로 시계열을 training 기간, Validation 기간, Test 기간으로 분할<br>
계절이 있는 경우 일반적으로 각 기간에 전체 계절 수가 포함되도록 해야함<br>
<br>

test 데이터를 사용하여 retrain할 수도 있음<br>
\> test 데이터가 현재 시점과 가장 가까운 데이터라서 미래 가치를 결정하는 데 있어 가장 강력한 신호가 될 수 있음<br>
\> test 세트를 포기하는 것이 일반적. training과 validation 기간을 사용하요 훈련하면 test 세트가 미래가 되는 것<br>
<br>

#### roll-forward partitioning 
짧은 훈련 기간부터 시작하여 점차적으로 그것을 늘려감 (한 번에 하루 또는 한 번에 1 주일 씩)<br>
각 반복마다 훈련 기간에 모델을 학습<br>
검증 기간에서 다음 날 또는 다음 주를 예측하는 데 사용<br>
<br>

### 모델 평가
모델과 기간이 있으면 모델을 평가 가능<br>
성과를 계산하기 위한 메트릭이 필요<br>

#### 오류 계산
errors = forecasts - actual<br>
모델의 예측 값과 평가 기간 동안의 실제 값의 차이<br>

#### MSE (mean squared error) - 평균 제곱 오차
예측 성능을 평가<br>
mse = np.square(errors).mean()<br>
오류를 제곱하고 그 평균을 계산하는 평균 제곱 오차 또는 MSE
부정적인 값을 없애기 위해 제곱 수행<br>
큰 오류가 잠재적으로 위험하고 작은 오류보다 훨씬 많은 비용이 들면 mse를 선호<br>
```
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
```

#### RMSE (root means squared error)
rmse = np.sqrt(mse)<br>
오류 계산의 평균이 원래 오류와 동일한 규모로 되도록하려면 제곱근을 얻음<br>

#### MAE (mean absolute error) - 평균 절대 오차
mae = np.abs(errors).mean()<br>
네거티브를 없애기 위해 제곱하는 대신 절대 값을 사용<br>
이득이나 손실이 오류의 크기에 비례한다면 mae가 더 나음<br>
```
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
```
MAE 측정<br>

#### MAPE (mean absolute percentage error)
mape = np.abs(errors / x_valid).mean()<br>
절대 오차와 절대값 사이의 평균 비율<br>

### 예측 방법 : Moving Average - 이동 평균
이동 평균 (MA)을 계산<br>
<br>

MV 정의<br>
```
def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)
```
이전 30점의 평균 \> 좋은 스무딩 효과를 제공<br>
```
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
```
mse, mae 결과 = Naive forecasting 보다 나쁨<br>
<br>

미래에 대해 예측하려는 기간에 따라 실제로 Naive forecasting보다 더 나빠질 수 있음 (mae = 7.14)<br>
이를 방지하는 한 가지 방법은 차이점 분석(differencing)이라는 기술을 사용하여 시계열에서 추세와 계절성을 제거하는 것<br>
<br>

시간 t에서의 데이터와 그 이전의 365일 데이터 간의 차이<br>
```
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show() # 계절성이 사라짐
```
```
# 이동 평균 계산
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
# 계절성의 영향을 받지 않는 비교적 부드러운 이동 평균
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()
```

시계열 자체를 연구하는 대신 시간 T의 값과 이전 기간의 값의 차이를 연구<br>
  - 1년 전의 경우 시간 T에서 365를 빼면 추세와 계절성이 없는 difference time series을 얻을 수 있음 \> MA를 사용하여 예측<br>
original time series에 대한 최종 예측을 얻으려면 시간 T에서 365를 뺀 값을 다시 추가<br>
```
# 과거 뺀 값을 다시 추가
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
# 꽤 좋은 예측을 볼 수 있음
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()
```
검증 기간에 MAE측정하면 약 5.8이 됨<br>
Naive forecasting보다 약간 좋지만 엄청나게 좋은 것은 아님<br>
\> 과거 잡음을 제거함으로써 이러한 예측값을 개선할 수 있음<br>
```
# 노이즈 제거
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
```
\> 검증 기간의 MSE = 4.5<br>
전체 오차를 측정하면 수치는 육안 검사에 동의하고 오류율이 더 향상되었음<br>
실제로 시리즈가 생성되기 때문에 완벽한 모델이 잡음으로 인해 약 4의 평균 절대 오차(MAE)를 줄 것이라고 계산할 수 있음<br>
<br>

[Exercise 13 in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/Week%201%20Exercise%20Question.ipynb)<br>
<br>

[Exercise 13_answer in colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/Week%201%20Exercise%20Answer.ipynb)<br>
<br>


