import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import torch
from torch import nn, optim
import time
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
import random

import numpy as np

from model import MyModel
from util import CSVDataset, CSV_noniid_Dataset, calc_performance, CSV24DDataset, CSV79DDataset, Dirichlet_Dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_data(args, device):
    csv_file_folder = r'CSV-03-11/03-11-V2/kd_fl_split_by_label/X_train_Y_train'
    if not args.noniid:
        train_loaders = [
            DataLoader(
                CSVDataset(os.path.join(csv_file_folder, "client_" + str(i) + ".csv"), device, randomly_drop_frac=False),
                batch_size=args.batch, shuffle=True)
            for i in range(0, args.clientnum)
        ]
    else:
        train_loaders = Dirichlet_Dataset(csv_file_folder, 0, 3, args)
        train_loaders.extend(Dirichlet_Dataset(csv_file_folder, 1, 3, args))
        train_loaders.extend(Dirichlet_Dataset(csv_file_folder, 2, 3, args))
    
    if args.drop_label:
        drop_labels = []
        if args.drop_strategy == 'random':
            labels_ls = list(range(args.classnum)) + list(range(args.classnum))  # because the client num larger than the label num, use this for easy sample.
            drop = random.sample(labels_ls, args.clientnum)
            drop_labels = [[i] for i in drop]
            print(f"For randomly sample non-iid dataset, each client drop the label by the sequence of {drop_labels}.")
            train_loaders = [
                DataLoader(
                    CSV_noniid_Dataset(os.path.join(csv_file_folder, "client_" + str(i) + ".csv"), drop_labels[i], device),
                    batch_size=args.batch, shuffle=True)
                for i in range(0, args.clientnum)
            ]

        elif args.drop_strategy == 'serial':
            labels_ls = list(range(args.classnum)) + list(range(args.classnum))
            drop_labels = [[i] for n,i in enumerate(labels_ls) if n<args.clientnum]
            print(f"For serially sample non-iid dataset, each client drop the label from small to large, according to the sequence of {drop_labels}.")
            train_loaders = [
                DataLoader(
                    CSV_noniid_Dataset(os.path.join(csv_file_folder, "client_" + str(i) + ".csv"), drop_labels[i], device),
                    batch_size=args.batch, shuffle=True)
                for i in range(0, args.clientnum)
            ]

    


    validation_loaders = [
        # DataLoader(CSVDataset(r'CSV-03-11/03-11/fl_split_by_label/X_test_Y_test.csv', device),
        #            batch_size=args.batch, shuffle=False)
#        DataLoader(CSV24DDataset(r'CSV-03-11/03-11/fl_split_by_label/X_test_Y_test.csv', device),
#                   batch_size=args.batch, shuffle=True)
       DataLoader(CSVDataset(r'CSV-03-11/03-11-V2/fl_split_by_label/X_test_Y_test.csv', device, randomly_drop_frac=False),
                  batch_size=args.batch, shuffle=True)
        for i in range(0, args.clientnum)
    ]

    test_loaders = [
        DataLoader(CSVDataset(r'CSV-03-11/03-11-V2/fl_split_by_label/X_test_Y_test.csv', device, randomly_drop_frac=False),
                   batch_size=args.batch, shuffle=False)
    ]

    return train_loaders, validation_loaders, test_loaders


def L1_Regularization(model):
    L1_reg = 0
    for param in model.parameters():
        L1_reg += torch.sum(torch.abs(param))

    return L1_reg


def train(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        # x = x.to(device).float()
        # y = y.to(device).long()

        output = model(x)
        loss = loss_func(output, y)
        # loss = loss_fun(output, y) + L1_Regularization(model) * args.wdecay

        loss.backward()
        loss_all += loss.item()

        optimizer.step()

    return loss_all / len(train_iter)


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            # data = data.to(device).float()
            # target = target.to(device).long()

            output = model.predict(data)

            pred = output
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_error = (total - correct) / total

    return test_error


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def public_dataset():
    # DONE: for convinence, temperarily choose client 0 as public dataset
    # TODO: use other data-free methods like Generator to distill knowledge
    idx = 0
    csv_file_folder = r'CSV-03-11/03-11/fl_split_by_label/X_train_Y_train'
    proxy_dataloader = DataLoader(
        CSVDataset(os.path.join(csv_file_folder, "client_" + str(idx) + ".csv"), device, randomly_drop_frac=False),
        batch_size=args.batch, shuffle=False)
    
    return proxy_dataloader


def proxy_dataset():
    # randomly choose parts of the sample from each client, view these as the proxy dataset
    # serve for the following ensemble KD
    csv_file = r'CSV-03-11/03-11-V2/kd_fl_split_by_label/X_train_Y_train/proxy_data.csv'
    proxy_dataloader = DataLoader(
        CSVDataset(csv_file, device, randomly_drop_frac=False),
        batch_size=args.batch, shuffle=True)
    
    return proxy_dataloader
    

def distillation_communicate(args, server_model, models, client_weights, test_loaders):
    # Average the clients' models and update the server model
    for key, value in server_model.state_dict().items():
        temp = torch.zeros_like(value)
        for idx, client_weight in enumerate(client_weights):
            temp += client_weight * models[idx].state_dict()[key]
        server_model.state_dict()[key].data.copy_(temp)
        
    # test after average aggregation, currently only gots one test_loader
    for _, test_loader in enumerate(test_loaders):
        test_error = test(server_model, test_loader)
        print(f' Server model by simply average aggregation | Error Rate: {test_error * 100.:.2f} %.')
        
        # throught the proxy dataset, test the performance of each client, get the confidence of various clients for following distillation
        client_accs = list()
        for model in models:
            client_acc = 1 - test(model, test_loader)
            client_accs.append(client_acc)
        client_accs = torch.tensor(client_accs)
        distill_weights = F.softmax(client_accs / args.weights_tmp, dim=0).to(device=args.device)
        
        print(client_accs)
        print(distill_weights)
        
    # when test error is already low, do not use KD, just simple average aggregation
    if test_error*100 < 2:
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        print(f'The server side model is already good by average aggregation. No need to use knowledge distillation for tuning.')
        return server_model, models

    # use proxy dataset to generate knowledge
    # proxy_dataloader = public_dataset()
    proxy_dataloader = proxy_dataset()
    KL_loss = nn.KLDivLoss(reduction='sum', log_target=True)
    kd_optimizer = optim.Adam(params=server_model.parameters(), lr=args.lr_kd)
    
    print(f"Set temperature to {args.tmp}, learning rate for KL divergance to {args.lr_kd}.")
    
    for epoch in range(args.distill_epochs):
        for step, data in enumerate(proxy_dataloader):
            x, y = data[0], data[1]
            # calculate teacher knowledge
            with torch.no_grad():
                for idx, model in enumerate(models):
                    logists = torch.unsqueeze(model(x), dim=0)
                    if idx == 0:
                        soft_labels = logists
                    else:
                        soft_labels = torch.concat([soft_labels, logists], dim=0)
            # add weights for each clients
            soft_labels.transpose_(0, 1)
            # print(distill_weights.shape, soft_labels.shape)
            knowledge = torch.matmul(distill_weights, soft_labels)
            print(knowledge.shape)

            # navie average client weights
            # knowledge = sum(soft_labels) / len(soft_labels)

            # calculate student logits
            output = server_model(x)

            T = args.tmp
            soft_outputs_log = F.log_softmax(output / T, dim=1)
            knowledge_log = F.log_softmax(knowledge / T, dim=1)
            soft_loss = KL_loss(soft_outputs_log, knowledge_log)
            
            print(f"The knowledge distillation loss is: {soft_loss}.")
            print(f"Some example knowledge is like: {F.softmax(knowledge[0:2] / T, dim=1), y[0:2]}.")

            kd_optimizer.zero_grad()
            soft_loss.backward()
            kd_optimizer.step()
        
            
    '''
    with torch.no_grad():
        for idx, model in enumerate(models):
            for step, data in enumerate(proxy_dataloader):
                x, y = data[0], data[1]
                logists = model(x)
                if step==0:
                    soft_labels = logists
                else:
                    soft_labels = torch.concat((soft_labels, logists), dim=0)
            soft_labels = torch.unsqueeze(soft_labels, 0)
            # shape: [len(models), len(data), len(labels)]
            if idx==0:
                all_model_soft = soft_labels
            else:
                all_model_soft = torch.concat((all_model_soft, soft_labels), dim=0)
        
    knowledge = torch.mean(all_model_soft, dim=0)   # shape: [len(data), len(labels)]
    # knowledge = all_model_soft[2]
        
    # calculate student outputs
    for step, data in enumerate(proxy_dataloader):
        x, y = data[0], data[1]
        output = server_model(x)
        if step==0:
            soft_outputs = output
        else:
            soft_outputs = torch.concat((soft_outputs, output), dim=0)  # shape: [len(data), len(labels)] 
    
    # calculate Kullback-Leibler divergence loss
    print(f"Set temperature to {args.tmp}, learning rate for KL divergance to {args.lr_kd}.")
    T = args.tmp
    KL_loss = nn.KLDivLoss(reduction='sum', log_target=True)
    soft_outputs_log = F.log_softmax(soft_outputs / T, dim=1)
    knowledge_log = F.log_softmax(knowledge / T, dim=1)
    soft_loss = KL_loss(soft_outputs_log, knowledge_log)
    
    print(f"The knowledge distillation loss is: {soft_loss}.")
    print(f"Some example knowledge is like: {F.softmax(knowledge[-10:] / T, dim=1)}.")
    
    # optimize the student network using only soft_loss
    kd_optimizer = optim.Adam(params=server_model.parameters(), lr=args.lr_kd)
    kd_optimizer.zero_grad()
    soft_loss.backward()
    kd_optimizer.step()
    '''
    
    # dispatch the server model to all clients
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    # decay the temperature and learning rate of KD
    args.lr_kd *= 0.95
    # if args.tmp>1.0:
    #     args.tmp = (args.tmp-1.0) * 0.95 + 1.0
    # else:
    #     args.tmp = 1.0 - (1.0-args.tmp) * 0.95

    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:', device, '\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    # parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--wdecay', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--iters', type=int, default=30, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=3,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedavg', help='fedavg')
    parser.add_argument('--save_path', type=str, default='checkpoint/mnist', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    parser.add_argument('--clientnum', type=int, default=9, help='client number')
    parser.add_argument('--setnum', type=int, default=10, help='set number per client has')
    parser.add_argument('--classnum', type=int, default=7, help='class num') 

    parser.add_argument('--seed', type=int, default=46, help='random seed')
    
    parser.add_argument('--KD', action='store_true', help='whether to use the ensemble knowledge distillation during the aggregation')
    parser.add_argument('--distill_epochs', type=int, default=1, help='epochs for the distillation process')
    parser.add_argument('--weights_tmp', type=int, default=0.5, help='temperature for the clients weight balance')
    parser.add_argument('--tmp', type=int, default=1.5, help='temperature for the KD, typically initialize T>=1')
    parser.add_argument('--lr_kd', type=int, default=0.001, help='learning rate for the knowledge distillation')

    parser.add_argument('--noniid', action='store_true', help='noniid sampling')
    parser.add_argument('--alpha', type=float, default=1, help='dirichlet distribution for dataset sample')
    parser.add_argument('--drop_label', action='store_true', help='whether to drop one label for each client') 
    parser.add_argument('--drop_strategy', choices=['random', 'serial'], help='how to drop label') 

    args = parser.parse_args()
    
    args.device = device
    print(args)
    
    wandb.init(sync_tensorboard=False,
               project="FL_KD",
               job_type="CleanRepo",
               config=args
               )

    setup_seed(args.seed)

    exp_folder = 'mnist_fedavg'

    args.save_path = os.path.join(args.save_path, exp_folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path,
                             args.mode + 'client' + str(args.clientnum) + 'sets' + str(args.setnum) + 'seed' + str(
                                 args.seed) + str(args.noniid))

    # server model and ce loss
    server_model = MyModel(input_dim=82, output_dim=args.classnum).to(device)
    loss_fun = loss_func = nn.CrossEntropyLoss().cuda()
    # loss_fun = loss_func = nn.NLLLoss().cuda()

    # prepare the data
    train_loaders, validation_loaders, test_loaders = prepare_data(args, device)

    print('\nData prepared, start training...\n')

    # federated setting
    client_num = args.clientnum
    clients = ['client' + str(_) for _ in range(1, client_num + 1)]
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        this_test_error = []
        for test_idx, test_loader in enumerate(test_loaders):
            test_error = test(server_model, test_loader)
            this_test_error.append(test_error)
            print(' {:<8s}| Error Rate: {:.2f} %'.format(clients[test_idx], test_error * 100.))
        print('Best Test Error: {:.2f} %'.format(100. * sum(this_test_error) / len(this_test_error)))

        exit(0)

    best_test_error = 1.
    training_loss_log = []
    error_rate_log = []

    lr = args.lr
    # start training
    for a_iter in range(args.iters):
        # record training loss and test error rate
        this_test_error = []
        this_train_loss = []

        optimizers = [optim.Adam(params=models[idx].parameters(), lr=lr) for idx in range(client_num)]
        lr *= 0.95

        for wi in range(args.wk_iters): 
            print("============ Train epoch {} ============".format(wi + 1 + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss = train(args, model, train_loader, optimizer, loss_fun, client_num, device)
                print(' {:<8s}| Train Loss: {:.4f}'.format(clients[client_idx], train_loss))

                this_train_loss.append(train_loss)
                
        # abalation
        for ii in range(9):
            print(f"********** Client {ii} ************")
            calc_performance(models[ii], validation_loaders[0])

        # aggregation
        if args.KD:
            server_model, models = distillation_communicate(args, server_model, models, client_weights, test_loaders)
        else:
            server_model, models = communication(args, server_model, models, client_weights)
        
        # start testing
        for test_idx, test_loader in enumerate(validation_loaders):
            test_error = test(server_model, test_loader)
            this_test_error.append(test_error)
            if args.KD:
                print(' Server model after knowledge distillation | Error Rate: {:.2f} %'.format(test_error * 100.))
            else:
                print(' {:<8s} | Error Rate: {:.2f} %'.format(clients[test_idx], test_error * 100.))
            break
            
        wandb.log({'Error Rate/Server': test_error}, step=a_iter)

        # since we only have one test/validation set
        calc_performance(server_model, validation_loaders[0])

        # error rate after this communication
        this_test_error = sum(this_test_error) / len(this_test_error)
        if this_test_error < best_test_error:
            best_test_error = this_test_error

            # Save checkpoint
            print(' Saving checkpoints to {}'.format(SAVE_PATH))
            torch.save({
                'server_model': server_model.state_dict(),
                'a_iter': a_iter,
            }, SAVE_PATH)

        # Best Validation Error Rate
        print(' Best Validation Error Rate: {:.2f} %, Current Validation Error Rate: {:.2f} %\n'.format(
            best_test_error * 100.,
            this_test_error * 100.
        ))

        training_loss_log.append(sum(this_train_loss) / len(this_train_loss))
        error_rate_log.append(this_test_error)

        if not os.path.exists(os.path.join('logs/mnist_fedavg', args.mode)):
            os.makedirs(os.path.join('logs/mnist_fedavg', args.mode))

    print('Start final testing\n')
    checkpoint = torch.load(SAVE_PATH)
    server_model.load_state_dict(checkpoint['server_model'])
    this_test_error = []
    for test_idx, test_loader in enumerate(test_loaders):
        test_error = test(server_model, test_loader)
        this_test_error.append(test_error)
    print('Best Server-side Test Error: {:.2f} %'.format(100. * sum(this_test_error) / len(this_test_error)))

    error_rate_log.append(sum(this_test_error) / len(this_test_error))
    # save record
    np.savetxt(os.path.join('logs/mnist_fedavg', args.mode, 'client' + str(args.clientnum) +
                            'sets' + str(args.setnum) + 'seed' + str(args.seed) + str(args.noniid) + 'train_loss.txt'),
               training_loss_log, newline="\r\n")
    np.savetxt(os.path.join('logs/mnist_fedavg', args.mode, 'client' + str(args.clientnum) +
                            'sets' + str(args.setnum) + 'seed' + str(args.seed) + str(args.noniid) + 'error_rate.txt'),
               error_rate_log, newline="\r\n")
