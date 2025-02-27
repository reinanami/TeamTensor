import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub

'''
**************************************** PLEASE READ ****************************************************
1. If you want to download this Kaggle training data: https://www.kaggle.com/c/111-1-ntut-dl-app-hw4/code
2. Put it in dataset/human_detection_in_drone_videos
3. Download this Kaggle training data as well: https://www.kaggle.com/datasets/rgbnihal/c2a-dataset
4. Put it in dataset/human_deection_disaster_scenarios
5. Modify the code as needed!

# Uncomment if placing dataset from Kaggle from downloaded data
api = KaggleApi
api.authenticate()
api.dataset_download_files('dataset/human_detection_in_drone_videos', path='./data', unzip=True)

# Uncomment to use the Kaggle method

path = kagglehub.dataset_download("rgbnihal/c2a-dataset")

print("Path to dataset files:", path)

*********************************************************************************************************
'''

# Kaggle
'''
< PASTE KAGGLE STUFF FROM ABOVE HERE > 

'''

# Load dataset
filePath = f'{path}/dataset.csv'
data = pd.read_csv(filePath)

# Convert dataset to tensor format

''' This is defaulted please modify as needed! '''

states = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)

actions = torch.tensr(data.iloc[:, -1].values, dtype=torch.long)


'''
**************************************** NEURAL NETWORKS ************************************************
'''
class NN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(NN, self).__init__()

        self.L1 = nn.Linear(inputSize, outputSize)
        self.L2 = nn.Linear(inputSize, outputSize)
        self.L3 = nn.Linear(inputSize, outputSize)

    def forward(self, x):

        # The activation functions can be modified to fit the microcontrollers
        # Use Sigmoid if that's better

        x = F.relu(self.L1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.L2(x))
        x = F.dropout(x, p=0.5)
        x = self.L3(x)

        return x
    
'''
*********************************************************************************************************
'''
'''
************************* SIZE, PARAMETERS, LOSS FUNCTION, OPTIMIZATION *********************************
'''
inputSize = states.shape[1]

actionSize = len(actions.unique())

model = NN(inputSize, actionSize)

criterion = nn.CrossEntropyLoss() # Change to MSE if that's better

optimizer = optim.Adam(model.parameters(), lr=0.01) # Modify the learning rate

# IMPORTANT: Please research ADAM vs SGD because ADAM works on CPU, CUDA, MPS!

'''
*********************************************************************************************************
'''

'''
****************************************** TRAINING THE MODEL *******************************************
'''
numEpochs = 20

for epoch in range(numEpochs):
    optimizer.zero_grad()
    outputs = model(states)
    loss = criterion(outputs, actions)
    loss.backward()
    optimizer.step()

    # This is the logging frequency. I have no experinece with this, please modify
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{numEpochs}], Loss: {loss.item():4.4f}')

'''
*********************************************************************************************************
'''

# Test model on a new state!

newstate = torch.tensor(np.random.rand(1, inputSize), dtype=torch.float32)
predictedAction = torch.argmax(model(newState)).item()

# You can modify this to test multiple states
print(f'Predicted action: {predictedAction}')  