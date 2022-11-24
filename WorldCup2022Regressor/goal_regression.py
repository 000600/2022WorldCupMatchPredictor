# Imports
from re import X
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get data
df = pd.read_csv('imatches.csv')
df = pd.DataFrame(df)

# Remove unhelpful columns
df = df.drop(['date', 'home_team_continent', 'away_team_continent', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'tournament', 'city', 'country', 'neutral_location', 'shoot_out', 'home_team_result', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score'], axis = 1)

# Convert the dataframe into a list
df_list = []
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0])):
    rows.append(df.iloc[row][point])
  df_list.append(rows)

x = []
y = []

# Loop through all past world cup games
for game in df_list:
  # Input is a list: [country1 FIFA ranking, country2 FIFA ranking]
  x.append([game[2], game[3]])

  # Output is list: [country1's goals, country2's goals] 
  y.append([game[4], game[5]])


# Get and split the data
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size = 0.2, random_state = 1)

# Get input shape
input_shape = len(x[0])

# Create Adam optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Input layer
model.add(BatchNormalization())
model.add(Dense(2, activation = 'relu', input_shape = [input_shape]))

# Hidden layers
model.add(Dense(1, activation = 'relu'))

# Output layer
model.add(Dense(2, activation = 'relu')) # No activation function because the model is a regression algorithm

# Compile model
model.compile(optimizer = opt, loss = 'mse', metrics = ['accuracy'])
early_stopping = EarlyStopping(min_delta = 0.01, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 100
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test), callbacks = [early_stopping])

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict(x_train, verbose = 0)[index]
actual_value = y_train[index]

print(f"\nModel's Prediction: First Country's Score - {prediction[0]}, Second Country's Score - {prediction[1]} | Actual Results:  First Country's Score - {actual_value[0]}, Second Country's Score - {actual_value[1]}")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# Define a function to use the model to make real predictions
def display_predictions(country1, country2, fifa_ranking1, fifa_ranking2):
  pred = model.predict([[fifa_ranking1, fifa_ranking2]])
  score1 = round(pred[0][0])
  score2 = round(pred[0][1])
  text = f"\nPredicted Score in a Match Between {country1} (FIFA Ranking of {fifa_ranking1}) and {country2} (FIFA Ranking of {fifa_ranking2}): {country1} ({score1}), {country2} ({score2})"
  return text

print(display_predictions("Argentina", "Mexico", 3, 13))
