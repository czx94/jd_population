# jd_population
JDD2018

## exp1

- result = sample_result_mod_7[(274 + d) % 7] * 0.9 + total_result_mod_7[(274 + d)]
- 0.3139

## exp2

- result = sample_result_mod_7[(274 + d) % 7] * 0.5 + total_result_mod_7[(274 + d) % 7] * 0.1 + sample_result_mod_30[(274 + d) % 30] * 0.4
- 0.3136

## exp3

- result = sample_result_mod_30[(274 + d) % 30]
- 0.2332

## exp4

- result = sample_result_mod_7[(274 + d) % 7] 
- 0.2185

## exp5

- result = ARIMA(1,1,5), offset = 0
- 0.1425

## exp6

- result = ARIMA(1,1,6), offset = 1
- 0.1430

## exp7

- result = ARIMA(1,1,6), offset = 0
- 0.1510

## exp8

- result = ARIMA(1,1,5), offset = 1
- 0.1341