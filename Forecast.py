# importing libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


if True:  # noqa: E402
    import matplotlib.dates as md
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    from functools import partial
    from keras import Input                                                 # noqa
    from keras.src.layers import Dense, LSTM, Dropout                       # noqa
    from keras.src.models import Sequential                                 # noqa
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels. stats.diagnostic import het_white

    from util.tools import init_environment, save_object, load_object       # noqa


# Preparing the environment
print = partial(print, end='\n\n\n')

CRED = '\033[42m'
CEND = '\033[0m'

rows, cols = 3, 4

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})
pd.set_option('display.max_columns', None)
# pd.set_option('expand_frame_repr', False)  # To expand the print without

file_name_model = 'data/model.pkl'
init_environment()


# Loading the data
print(CRED + 'Loading the data' + CEND)
all_data = pd.read_csv('data/stock.csv', parse_dates=['date'])
# all_data['date'] = all_data['date'].dt.date
print(all_data.head())
print(all_data.info())
print(all_data.describe(include='all', percentiles=[]).T)
print(all_data.sname.value_counts())


# We are going to make predictions only for gspc index
print(CRED + 'Taking a look at GSPC Index' + CEND)
gspc_data = all_data[all_data.sname == '^GSPC'].reset_index(drop=True)
gspc_data = gspc_data.drop('sname', axis=1)
print(gspc_data.describe(include='all', percentiles=[]).T)


# EDA
print(CRED + '/Making an EDA' + CEND)
plt.figure()
plt.subplot(rows, cols, 1)
plt.plot(gspc_data['date'], gspc_data['value'])
plt.xlabel('date')
plt.ylabel('value')
plt.title('Company share price')


# Extract the growth trend with rolling window, with the rounding period equal
# to month
print(CRED + 'Resampling the data' + CEND)
gspc_resampled = gspc_data.resample('ME', on='date').mean()

ax = plt.subplot(rows, cols, 2)
gspc_resampled.plot(ax=ax, legend=False)
plt.xlabel('date')
plt.ylabel('value')
plt.title('Resampled plot')


# Review autocorrelation and partial autocorrelation to determine direct and
# transitive long-term dependencies between values
print(CRED + 'Reviewing the partial autocorrelation' + CEND)
ax = plt.subplot(rows, cols, 3)
sm.graphics.tsa.plot_pacf(gspc_data['value'].values, lags=10, method="ywm", ax=ax)  # noqa
plt.xlabel('Lag(days)')
plt.ylabel('Correlation')

print(CRED + 'Reviewing the autocorrelation' + CEND)
ax = plt.subplot(rows, cols, 4)
sm.graphics.tsa.plot_acf(gspc_data['value'].values, lags=1000, ax=ax)
plt.xlabel('Lag(days)')
plt.ylabel('Correlation')
# There is not stationary trend and time serie has long-term recurrent
# dependencies (autocorrelation plot).


# Getting the first differences plot (getting residuals)
print(CRED + 'Reviewing the residuals' + CEND)
gspc_data_1_diff = gspc_data['value'].diff(periods=1).dropna()

ax = plt.subplot(rows, cols, 5)
gspc_data_1_diff.plot(ax=ax)
plt.xlabel('Days passed from Jan 2010')
plt.ylabel('value')
plt.title('Plot of first differences')


# Let's look at autocorrelation, partial autocorrelation and perform White's
# test to check homoscedasticity
# Evaluations required in Linear Regression Models
# (1) Means of residuals should be 0 --> ols_model.resid.mean()
# (2) Normallity of error terms --> plot the histogram of residuals
# (3) Linearity of variables --> sns.residplot
# (4) No heteroscedasticity --> Residuals are homoscedastic, sms.het_goldfeldquandt (𝑝𝑣𝑎𝑙>0.05)  # noqa
#                               het_white (pval>0.05) is no homscedastic
print(CRED + 'Reviewing the residuals partial autocorrelation' + CEND)
ax = plt.subplot(rows, cols, 6)
sm.graphics.tsa.plot_pacf(gspc_data_1_diff, lags=10, method="ywm", ax=ax)
plt.title('Residuals\nPartial Autocorrelation')
plt.xlabel('Lag(days)')
plt.ylabel('Correlation')

print(CRED + 'Reviewing the residuals autocorrelation' + CEND)
ax = plt.subplot(rows, cols, 7)
sm.graphics.tsa.plot_acf(gspc_data_1_diff, lags=10, ax=ax)
plt.title('Residuals\nAutocorrelation')
plt.xlabel('Lag(days)')
plt.ylabel('Correlation')

print(CRED + 'Reviewing the homoscedasticity' + CEND)
x = np.arange(len(gspc_data_1_diff))  # 4748
y = np.array(gspc_data_1_diff)
x = sm.add_constant(x)

ols_model = sm.OLS(y, x).fit()
white_test = het_white(ols_model.resid, ols_model.model.exog)
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']  # noqa
homoscedastic_test = dict(zip(labels, white_test))
print(*homoscedastic_test.items(), sep='\n')
if homoscedastic_test['Test Statistic p-value'] < 0.05:
    print('There is sufficient evidence to say there is heteroscedasticity in the data.')  # noqa
else:
    print('There is sufficient evidence to say there is homoscedasticity in the data.')  # noqa
# We can see that samples of differences series have 0 autocorrelation for all
# lags > 0 - they are linearly independent and we won`t be able to use linear
# model(ARIMA) to make a prediction (correlation can measure only linear
# dependencies - so these samples can have more complicated non-linear
# dependencies and have 0 correlation at the same time).
# White's test p_value is less than 0.05, so our data is heteroscedastic.
# As a result we can make a conclusion that it won't be usefull to use
# differences for making predictions.


# Using recurrent neural network
# RNN can be trained to detect latent long-term dependencies and forecast
# such a complicated time series.
# For the loss function we will use MSE - standart metric to solve regression
# problems.
# Isolating the data
print(CRED + 'Isolating the data' + CEND)
data = gspc_data.filter(['value'])
dataset = data.values
print(dataset)


# Scaling the data
print(CRED + 'Scaling the data' + CEND)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)


# Visualizing scaled data
print(CRED + 'Visualizing the scaled data' + CEND)
plt.subplot(rows, cols, 8)
plt.plot(scaled_data, label='Stocks data')
plt.title('Scaled stocks\nfrom Jan 2010 to Jan 2023')
plt.xlabel('Days')
plt.ylabel('Scaled value of stocks')


# Splitting the data into training and testing set
# We will use previous 180 values to forecast the next one
print(CRED + 'Splitting the data into training and testing set' + CEND)
train_data = scaled_data[0:4000]
x_train = []
y_train = []

for i in range(180, len(train_data)):
    x_train.append(train_data[i-180:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(f"""
Training set:
    x size >> {x_train.shape}
    y size >> {y_train.shape}
""")


test_data = scaled_data[4000-180:, :]
x_test = []
y_test = dataset[4000:, :]
for i in range(180, len(test_data)):
    x_test.append(test_data[i-180:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(f"""
Testing set:
    x size >> {x_test.shape}
    y size >> {y_test.shape}
""")


# Training the model
print(CRED + 'Training the model' + CEND)
init_environment()
# model = Sequential()
# model.add(Input(shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))  # noqa
# model.add(Dropout(0.3))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(units=150, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(units=200, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(units=250, return_sequences=False))
# model.add(Dense(units=100))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, batch_size=180, epochs=4, verbose=1)


# # Saving the model
# save_object(model, file_name_model)


# Getting back the model
model = load_object(file_name_model)


# Checking the performance
print(CRED + 'Evaluating the model' + CEND)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print('MSE between real data and predictions is: ', np.square(predictions-data[4000:]).mean(axis=0))  # noqa


# Reviewing the predictions
print(CRED + 'Reviewing the predictions' + CEND)
valid = pd.DataFrame({
    'y_true': y_test.ravel(),
    'y_predict': predictions.ravel(),
})
print(valid.head())


# Visualizing the predicted values
print(CRED + 'Visualizing the predicted value' + CEND)
plt.subplot(rows, cols, 9)
plt.plot(valid[['y_true', 'y_predict']])
plt.title('RNN Model')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(['True Values', 'Predictions'], loc='upper right')
# This model has a pretty good result and determines the trend
# well. This model can be improved by adding additional layers,
# more flexible parameter settings, and more regularizers.


# Data goes from 01.01.2010 to 01.01.2023
# Predicting new data(02.01.2021 - 01.02.2023)
print(CRED + 'Predicting on new data' + CEND)
new_data = scaled_data[-180:, :]
new_X = np.reshape(new_data, (1, new_data.shape[0], new_data.shape[1]))  # noqa
first_prediction = model.predict(new_X)

# Predicting next 30 months
new_data[0:179, :] = new_data[1:180, :]
new_data[179, :] = first_prediction[0, :]
for i in range(0, 30):
    new_X = np.reshape(new_data, (1, new_data.shape[0], new_data.shape[1]))  # noqa
    next_prediction = model.predict(new_X)
    new_data[0:179, :] = new_data[1:180, :]
    new_data[179, :] = next_prediction[0, :]
new_y = new_data[-31:]
new_y = scaler.inverse_transform(new_y.reshape(-1, 1))

# Visualizing the new predictions
new_predictions = pd.DataFrame({
    'date': pd.to_datetime(pd.date_range(start='2023-01-02', end='2023-02-01')),  # noqa
    'predicted': new_y.ravel()
})
print(f"""
New Data:
{new_predictions}
""")

ax = plt.subplot(rows, cols, 10)
plt.plot(new_predictions['date'], new_predictions['predicted'], color='red')
ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%d'))
plt.xlabel('date')
plt.ylabel('Predicted Values')
plt.title('New Values')


# All together
ax = plt.subplot(rows, cols, 11)
plt.plot(all_data['date'].values[-749:], valid['y_true'])
plt.plot(all_data['date'].values[-749:], valid['y_predict'])
plt.plot(new_predictions['date'], new_predictions['predicted'], color='red')
ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(['True Values', 'Predictions', 'New Values'], loc='upper right')
plt.title('Company share price')


# Display the plots
# plt.subplots_adjust(hspace=.3, wspace=.3)
plt.tight_layout()
plt.show()
plt.style.use('default')
