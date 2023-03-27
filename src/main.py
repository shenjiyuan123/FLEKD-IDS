import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import timeit

# fix random seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from model import MyModel


def calc_performance(loader):
    model.eval()
    predictions = []
    gt = []
    for i, data in enumerate(loader):
        inputs, labels = data
        outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        _, predicted = torch.max(F.softmax(outputs.data, dim=1), 1)
        predictions.append(predicted)
        gt.append(labels)
    gt = torch.cat(gt)
    gt = gt.cpu().numpy()
    predictions = torch.cat(predictions)
    predictions = predictions.cpu().numpy()
    print(confusion_matrix(gt, predictions))
    print(classification_report(gt, predictions, digits=4))
    # return confusion_matrix(loader.dataset.Y, predictions), classification_report(loader.dataset.Y, predictions)




from util import CSVDataset, CSV24DDataset, CSV79DDataset


# import pytorch datalaoder
from torch.utils.data import DataLoader
# import pytorch optim
from torch.optim import Adam

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("training on device: ", device)

# train MyModel with PyTorch Dataloader from csv file
# csv_file = r'../CSV-03-11/03-11/split_by_label/X_train_24d_Y_train.csv'
csv_file = r'../CSV-03-11/03-11/split_by_label/X_train_Y_train.csv'
# csv_file = r'../CSV-03-11/03-11/split_by_label/X_train_79d_Y_train.csv'
train_loader = DataLoader(CSVDataset(csv_file, device, max_limit_per_type=35), batch_size=512, shuffle=True)

# load test dataset
csv_file = r'../CSV-03-11/03-11/split_by_label/X_test_Y_test.csv'
# test_loader = DataLoader(CSV24DDataset(csv_file, device), batch_size=512, shuffle=True)
test_loader = DataLoader(CSVDataset(csv_file, device), batch_size=512, shuffle=True)
# test_loader = DataLoader(CSV79DDataset(csv_file, device), batch_size=512, shuffle=True)


epoch_total = 10
model = MyModel(input_dim=82, output_dim=9+1).to(device)
# model = MyModel(input_dim=24, output_dim=9+1).to(device)
# model = MyModel(input_dim=82, output_dim=9+1).to(device)
# model = MyModel(input_dim=78, output_dim=9+1).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
loss_func = nn.NLLLoss()
for epoch in range(epoch_total):
    start = timeit.default_timer()
    print("starting epoch: ", epoch, " at:", start)
    epoch_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("===============debug")
        # print(inputs, labels)
        # loss = loss_func(torch.log(outputs), labels)
        loss = loss_func(nn.LogSoftmax(dim=1)(outputs), labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch: {}/{}'.format(epoch+1, epoch_total), '| Step: {}'.format(i), '| Loss: {:.4f}'.format(epoch_loss))
            # print('Predicted:', outputs.data.max(1, keepdim=True)[1])
            # print('Actual:', labels.data)
            # print()
    print("epoch: ", epoch, " cost time:", timeit.default_timer() - start)
    print("performance on train")
    calc_performance(train_loader)
    print("performance on test")
    calc_performance(test_loader)


# save model
torch.save(model.state_dict(), 'model.pt')

