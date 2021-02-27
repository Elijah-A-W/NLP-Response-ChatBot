import json
from nltk_utils import tokenize, stem,bag_of_words
import numpy as np


import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet


with open('intents.json', 'r') as f:
    intents = json.load(f)


all_words = []
xy = []
tags = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_characters = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_characters]
#Using stricky unique words and arranging the words in a list format
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)


x_train = []
y_train = []

for (sentence_pattern, tag) in xy:
    #Putting all the pattern sentences into bag of words
    bag = bag_of_words(sentence_pattern, all_words)
    x_train.append(bag)
    #The target varibale are tags and in our case, the crossentropy doesnt need onehot encoding only needs the y-target which is the 
    label = tags.index(tag)
    y_train.append(label)

x_train  = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples



# print(input_size, len(all_words))


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


#adam optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        

        #foward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()


    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs},  loss={loss.item():}')   

print(f'final loss, loss={loss.item():.4f}')    

data = {

    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training is complete, and the file has been saved to{FILE}")