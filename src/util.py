from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class CSVDataset(Dataset):
    def __init__(self, filepath, device, max_limit_per_type = None):
        data = pd.read_csv(filepath)
        # group by label, and limit the number of samples per type
        if max_limit_per_type is not None:
            data = data.groupby('Label').apply(lambda x: x.sample(min(max_limit_per_type, len(x))))
        self.X = data.drop('Label', axis=1).values
        self.Y = data['Label'].values
        self.X = torch.from_numpy(self.X).type(torch.float32).to(device)
        self.Y = torch.from_numpy(self.Y).type(torch.long).to(device)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # return (torch.tensor(self.X[idx], dtype=torch.float32),
        #         torch.tensor(self.Y[idx], dtype=torch.long))
        return self.X[idx], self.Y[idx]


class CSV79DDataset(CSVDataset):
    def __init__(self, filepath, device, max_limit_per_type = None):
        raise Exception('Not implemented')
        super(CSV79DDataset, self).__init__(filepath, device)
        data = pd.read_csv(filepath)
        data.drop(columns=["Source Port", "Protocol", "SimillarHTTP",  "Inbound"], inplace=True)
        data = data.values
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.X = torch.from_numpy(self.X).type(torch.float32).to(device)
        self.Y = torch.from_numpy(self.Y).type(torch.long).to(device)


class CSV24DDataset(CSVDataset):
    def __init__(self, filepath, device, max_limit_per_type = None):
        raise Exception('Not implemented')
        data = pd.read_csv(filepath)
        data = data[['ACK Flag Count',
'Average Packet Size',
'Destination Port',
'Flow Duration',
'Flow IAT Max',
'Flow IAT Mean',
'Flow IAT Min',
'Fwd Header Length',
'Fwd Header Length.1',
'Fwd IAT Max',
'Fwd IAT Mean',
'Fwd IAT Total',
'Fwd Packet Length Max',
'Fwd Packet Length Min',
'Fwd Packet Length Std',
'Fwd Packets/s',
'Init_Win_bytes_forward',
'Total Length of Fwd Packets', #'Length of Fwd Packets',
'Max Packet Length',
'Min Packet Length',
'min_seg_size_forward',
'Packet Length Std',
'Protocol',
'Subflow Fwd Bytes', 'Label']]
        data = data.values
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.X = torch.from_numpy(self.X).type(torch.float32).to(device)
        self.Y = torch.from_numpy(self.Y).type(torch.long).to(device)


def calc_performance(model, loader):
    model.eval()
    predictions = []
    gt = []
    for i, data in enumerate(loader):
        inputs, labels = data
        predicted = model.predict(inputs)
        predictions.append(predicted)
        gt.append(labels)
    gt = torch.cat(gt)
    gt = gt.cpu().numpy()
    predictions = torch.cat(predictions)
    predictions = predictions.cpu().numpy()
    print(confusion_matrix(gt, predictions))
    print(classification_report(gt, predictions, digits=4))
    # return confusion_matrix(loader.dataset.Y, predictions), classification_report(loader.dataset.Y, predictions)

