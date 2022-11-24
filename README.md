# 2022 World Cup Match Predictor

## The Neural Network

This neural network predicts the score of two national soccer/football teams based on their respective FIFA ranking positions. The model takes an input list consisting of [*FIFA ranking of country 1*, *FIFA ranking of country 2]. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 and multiple dropout layers to prevent overfitting. The model has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 64 input neurons and a ReLU activation function)
- 4 Hidden layers (3 with 128 neurons and one with 256 neurons, each with a ReLU activation function)
- 2 Dropout layers (one after the first hidden layer and one after the last hidden layer, each with a dropout rate of 0.3)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download&select=heart.csv. Credit for the dataset collection goes to **Naman Manchanda**, **Durgance Gaur**, **Fahad Mehfooz** and others on *Kaggle*. It describes whether or not a heart attack is likely (encoded as 0 or 1) based on multiple factors, including:
- Age
- Number of major vessels (0-3)
- Type of chest pain (0 : typical angina, 1 : atypical angina, 2 :  non-anginal pain, 3 : asymptomatic) 
- Cholesterol
- Maximum achieved heart rate

A full description of all included features can be found on the dataset's webpage. Note that the initial dataset is biased (this statistic can be found on the data's webpage); it contains a higher representation of heart attack cases (encoded as 1's in this model) than non-heart attack cases (encoded as 0's in this model). This issue is addressed within the classifier file using  Imbalanced-Learn's **SMOTE()**, which oversamples the minority class within the dataset. Also, note that the data is scaled with Scikit-Learn's **StandardScaler()** before being fed into the model. 

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- Imbalanced-Learn's Website: https://imbalanced-learn.org/stable/about.html
- Imbalanced-Learn's Installation Instructions: https://pypi.org/project/imbalanced-learn/

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
