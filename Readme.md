# Machine Learning - Time Series Forecasting
Sample code to make forecasting given a time series.

Features:
- Time Series
- Forecasting
- LSTM
- RNN

## Run ML model
```
python Forecast.py
```

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/TimeSeriesForecasting.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Others
- Proyect in GitHub: https://github.com/jacesca/DataAnomalyDetection
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Commands to install TensorFlow (updated on Apr 26, 2024)
**Caution**: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin. 
Anything above 2.10 is not supported on the GPU on Windows Native

Source: https://www.tensorflow.org/install/pip#windows-native

I simply execute the following code:
```
pip install tensorflow
```
However the previous site, suggest to execute:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
```
Once installed you can verify it by executing:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Extra documentation
- [Disable Tensorflow debugging information](https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
- [Change default values of existing function's parameters](https://stackoverflow.com/questions/14608908/python-3-change-default-values-of-existing-functions-parameters)
- [Heteroscedasticity tests](https://nbviewer.org/github/jacesca/MIT-I/blob/a302a8a2f7604072911f097f3c684b4414053588/04%20-%20Machine%20Learning/Live%20Virtual%20Class%2001%20-%20Intro%20to%20Supervised%20Learning%20and%20Regression/Notebook%2003%20-%20Heteroscedasticity%20tests.ipynb)
- [Hospital Length of Stay (LOS) Prediction](https://nbviewer.org/github/jacesca/MIT-I/blob/a302a8a2f7604072911f097f3c684b4414053588/04%20-%20Machine%20Learning/Live%20Virtual%20Class%2002%20-%20Model%20Evaluation%20Cross%20Validation%20and%20Boostrapping/Notebook%2003%20-%20%28Original%29%20Hospital%20LOS%20PRediction%20Part%201.ipynb)
