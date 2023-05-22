from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, precision_recall_curve
import os
import numpy as np
import matplotlib.pyplot as plt

def Dirichlet_Dataset(dirpath, split_num, subset_num, args):
    n_clients = subset_num
    n_classes  = args.classnum
    alpha = args.alpha
    start = split_num*subset_num
    end   = (split_num+1)*subset_num
    for i in range(start, end):
        filepath = os.path.join(dirpath, "client_" + str(i) + ".csv")
        if i==start:
            data = pd.read_csv(filepath)
        else:
            data = pd.concat([data, pd.read_csv(filepath)])
    
    data.reset_index(drop=True, inplace=True)

    # use Dirichlet Process to sample, generate shape-like [n_classes, n_clients] array
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    labels = data['Label']
    class_idcs = [np.array(labels[labels == y].index.to_list()) for y in range(n_classes)]
    
    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]

    # 按照横轴labels迭代, 使用频数来split
    for c, fracs in zip(class_idcs, label_distribution):
        # i表示第i个client; idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            # 把每个client对应的样本索引相加
            client_idcs[i] += [idcs]
    
    # 把各个client位置里面的array concat到一个大array中
    # [[arr1, arr2 ..., arr_n], ...] => [n1, n2, ...]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(12,6))
    plt.hist([labels[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(start,end)], rwidth=0.5)
    plt.xticks(np.arange(n_classes), ['0','1','2','3','4','5','6'])
    plt.legend()
    plt.savefig('sample.pdf')
    
    '''
    # 展示不同client上的label分布
    plt.figure(figsize=(20, 6))  # 3
    label_distribution = [[] for _ in range(n_clients)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=['0','1','2','3','4','5','6'], rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                c_id for c_id in range(n_clients)])
    plt.ylabel("Number of samples")
    plt.xlabel("Client ID")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    '''

    print(data, client_idcs)
    train_loaders = list()
    for client in client_idcs:
        client_data = data.iloc[client]
        client_data.reset_index(drop=True, inplace=True)
        print(client_data.head())
        train_loaders.append(
            DataLoader(DFDataset(client_data, args.device), batch_size=args.batch, shuffle=True)
        )

    return train_loaders
        
         

class CSVDataset(Dataset):
    def __init__(self, filepath, device, randomly_drop_frac = 0.3, max_limit_per_type = None):
        data = pd.read_csv(filepath)
        # group by label, and limit the number of samples per type
        if max_limit_per_type is not None:
            data = data.groupby('Label').apply(lambda x: x.sample(min(max_limit_per_type, len(x))))
        if randomly_drop_frac:
            data = data.sample(frac=1)  # shuffle
            data = data.drop(data.sample(frac=randomly_drop_frac).index)
            print("After the random drop, the frequency of each label is:", data['Label'].value_counts())
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


class DFDataset(CSVDataset):
    def __init__(self, data, device):
        self.X = data.drop('Label', axis=1).values
        self.Y = data['Label'].values
        self.X = torch.from_numpy(self.X).type(torch.float32).to(device)
        self.Y = torch.from_numpy(self.Y).type(torch.long).to(device)

class CSV_noniid_Dataset(Dataset):
    def __init__(self, filepath, drop_label, device):
        data = pd.read_csv(filepath)
        noniid_data = data[~data['Label'].isin(drop_label)]
        self.X = noniid_data.drop('Label', axis=1).values
        self.Y = noniid_data['Label'].values
        self.X = torch.from_numpy(self.X).type(torch.float32).to(device)
        self.Y = torch.from_numpy(self.Y).type(torch.long).to(device)

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
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
    print(f"Precision: {precision_score(gt,predictions,average='weighted')}, Recall: {recall_score(gt,predictions,average='weighted')}, F1-score: {f1_score(gt,predictions,average='weighted')}")
    # return confusion_matrix(loader.dataset.Y, predictions), classification_report(loader.dataset.Y, predictions)

