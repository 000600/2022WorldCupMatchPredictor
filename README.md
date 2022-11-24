# 2022 World Cup Match Predictor

## The Neural Network

This neural network predicts the score of two national soccer/football teams based on their respective FIFA ranking positions. The model takes an input list consisting of the FIFA rankings of each national team ([[*FIFA_ranking_first_national_team*, *FIFA_ranking_second_national_team*]]) and outputs a corresponding list consisting of the score of the first inputted team and the score of the second inputted team ([[*score_first_national_team*, *score_second_national_team*]]). Since the model is a regression algorithm that predicts two outputs, the model uses a standard mean squared error loss function and has 2 output neurons. The model uses a standard Adam optimizer with a learning rate of 0.001 and has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 2 input neurons and a ReLU activation function)
- 1 Hidden layer (with 1 neuron with a ReLU activation function)
- 1 Output layer (with 2 output neuron and a ReLU activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022. Credit for the dataset collection goes to **Brenda_L**, **Rodrigo Alencar**, **SolomonYolo** and others on *Kaggle*. It contains a large amount of information pertaining to international matches, but the only factors used in this neural network are:
- Home team's FIFA rank
- Away team's FIFA rank
- Home team score
- Away team score
Naturally, a team's FIFA rank is not a perfect indicator of a match's score, and FIFA's rankings themselves are typically somewhat subjective. As a result there is not a perfect correlation between the inputs and the outputs the model is fed, so the network is by no means perfect — it tends to have an accuracy that hovers around 72%. Nonetheless, it is an interesting exercise in regression and sports prediction.

## Using the Model for Prediction
At the end of the file is a function **display_predictions**, which can be used to see what the model predicts the output of a match will be. To use the function, change the inputted parameters in the file to the first country, the second country, the first country's FIFA rank, and then the second country's FIFA rank. FIFA rankings can be found here (the model uses the far left column as inputs): https://www.fifa.com/fifa-world-ranking/men?dateId=id13792. Printing the output of the function will allow one to view the model's predicted scores for that match. 

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
