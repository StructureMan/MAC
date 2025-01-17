import json
import os
import sys

import torch
from clyent import color
from tqdm import tqdm
from src.folderconstants import output_folder
from src.parser import args
from src.models import *
from src.utils import *
from src.evaluate import *
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.io as sio
from src.data_utils import cacluateJaccard
import glob

device = "cuda:1"
import warnings
import pandas as pd
import time
import datetime

debug = False
warnings.filterwarnings("ignore")
# device = "cpu"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


# torch.autograd.set_detect_anomaly(True)
# from beepy import beep


class dataLoader(Dataset):
    def __init__(self, DataName="", modelName="", convertWindow="", stage="Train", item_dataSet="", ATT_mode="",
                 save_path=""):
        self.DataName = DataName
        self.modelName = modelName
        self.convertWindow = convertWindow
        self.stage = stage
        self.at = ATT_mode
        self.limitModel = ["MAC"]
        self.limitStackModel = []
        exception_models = ["MAC"]
        folder = os.path.join(output_folder, self.DataName)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        self.loader = []
        for file in ['train', 'test', 'labels']:
            if DataName == 'ASD': file = item_dataSet + file
            if DataName == 'SMD': file = item_dataSet + file
            if DataName == 'SMAP': file = item_dataSet + file
            if DataName == 'MSL': file = item_dataSet + file
            self.loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True)[20:])
        self.labes_ture = torch.tensor((np.sum(self.loader[2], axis=1) >= 1) + 0)
        # try:
        #     self.labes_ture = torch.tensor((np.sum(self.loader[2], axis=1) >= 1) + 0)
        # except:
        #     self.labes_ture = torch.tensor((np.sum(self.loader[2], axis=1) >= 1) + 0)
        self.labels_counts = torch.bincount(self.labes_ture)
        self.class_weights = 1.0 / self.labels_counts.float()
        self.sample_weights = self.class_weights[torch.tensor(self.labes_ture)]
        # 获取时域数据和频域数据
        self.frequency_train_data = self.loader[0][:, 0]
        self.frequency_test_data = self.loader[1]
        fft_data = self.faster_fourier_transform(self.frequency_train_data, 50)
        save_data = {
            "ori_data": self.frequency_train_data.tolist(),
            "time": [item for item in range(len(self.frequency_train_data.tolist()))],
            "fft_data": fft_data[1].tolist(),
            "frequency": fft_data[0].tolist()
        }

        file_name = save_path + "fft_data.json"
        os.makedirs(save_path, exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            f.close()
        if self.modelName in exception_models:
            self.convertWindow = convertWindow
        else:
            self.convertWindow = self.getDim()
        if self.modelName in self.limitModel:
            self.trainData = self.convertWindowPro(0)
            self.testData = self.convertWindowPro(1)
            self.labelData = self.convertWindowPro(2)
            self.adt_noise = np.random.normal(0.01, 0.3, size=self.trainData[0].shape)
        else:
            self.adt_noise = np.random.normal(0.01, 0.3, size=self.loader[0][0].shape)
        print("convertWindow", self.convertWindow)

    def faster_fourier_transform(self, signal, Fs):
        fft_signal = np.fft.fft(signal)
        n = len(signal)
        frequencies = np.fft.fftfreq(n, 1 / Fs)
        return frequencies, np.abs(fft_signal)

    def getDim(self):
        return self.loader[0].shape[1]

    def convertWindowPro(self, index):
        if self.modelName in self.limitModel:
            windows = []
            data = torch.Tensor(self.loader[index])
            for i, g in enumerate(self.loader[index]):
                if i >= self.convertWindow:
                    w = data[i - self.convertWindow:i]
                else:
                    w = torch.cat([data[0].repeat(self.convertWindow - i, 1), data[0:i]])
                if self.modelName in self.limitStackModel:
                    windows.append(w)
                else:
                    windows.append(
                        w if 'TranAD' in self.modelName or 'Attention' in self.modelName else w.contiguous().view(-1))
            return torch.stack(windows)

    def __len__(self):
        if self.stage == "Train":
            if self.modelName in self.limitModel:
                return len(self.trainData) - 1
            else:
                return len(self.loader[0]) - 1
        else:
            if self.modelName in self.limitModel:
                return len(self.testData) - 1
            else:
                return len(self.loader[1]) - 1

    def construct_t_adj(self, data):
        data = np.reshape(data, (self.convertWindow, -1))
        shape = data.shape
        distances = torch.pdist(data, p=2).to(device)
        adjacency_matrix = torch.zeros((shape[0], shape[0]), dtype=torch.float64).to(device)
        index = torch.triu_indices(shape[0], shape[0], offset=1).to(device)
        adjacency_matrix[index[0], index[1]] = distances.to(torch.float64)
        # adjacency_matrix = adjacency_matrix + adjacency_matrix.t()  # 保证邻接矩阵是对称的
        return adjacency_matrix.to(device)

    def construct_s_adj(self, data):
        data = np.reshape(data, (-1, self.convertWindow))
        shape = data.shape
        distances = torch.pdist(data, p=2).to(device)
        adjacency_matrix = torch.zeros((shape[0], shape[0]), dtype=torch.float64).to(device)
        index = torch.triu_indices(shape[0], shape[0], offset=1).to(device)
        adjacency_matrix[index[0], index[1]] = distances.to(torch.float64)
        # adjacency_matrix = adjacency_matrix + adjacency_matrix.t()  # 保证邻接矩阵是对称的
        return adjacency_matrix.to(device)

    def add_salt_and_pepper_noise(self, data, prob):
        data = data.cpu().detach().numpy()
        noise = np.random.choice([0, 1, 2], size=data.shape, p=[1 - prob, prob / 2, prob / 2])

        noisy_data = data.copy()
        noisy_data[noise == 1] = np.min(data)
        noisy_data[noise == 2] = np.max(data)

        return (noisy_data, noise)

    def add_gaussian_noise(self, data, mean=0, std_dev=1):
        data = data.cpu().detach().numpy()
        noise = np.random.normal(mean, std_dev, size=data.shape)

        noisy_data = data + noise

        return noisy_data

    def add_uniform_noise(self, data, low=0.01, high=0.3):

        try:
            noisy_data = data.cpu().detach().numpy() + self.adt_noise
        except:
            noisy_data = data + self.adt_noise

        return (noisy_data, self.adt_noise)

    def __getitem__(self, item):
        if self.stage == "Train":
            if self.at == "ADT":
                if self.modelName in self.limitModel:
                    ori_data = self.trainData[item]
                    # noise_salt_pepper = self.add_salt_and_pepper_noise(ori_data, prob=0.2)[0]
                    # noise_salt_pepper = self.add_gaussian_noise(ori_data, mean=0.3, std_dev=0.1)
                    noise_salt_pepper = self.add_uniform_noise(ori_data)[0]
                    # noise_salt_pepper = ori_data

                    return torch.FloatTensor(noise_salt_pepper).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(noise_salt_pepper, 50)[1]).to(device).to(torch.float64)
                else:
                    ori_data = self.loader[0][item]
                    noise_salt_pepper = self.add_uniform_noise(ori_data)[0]
                    # return [torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)]
                    return torch.FloatTensor(noise_salt_pepper).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(noise_salt_pepper, 50)[1]).to(device).to(torch.float64)
            elif self.at == "PGD":
                if self.modelName in self.limitModel:
                    ori_data = self.trainData[item]

                    # noise = self.add_salt_and_pepper_noise(ori_data, prob=0.2)[1]
                    # noise_salt_pepper = self.add_gaussian_noise(ori_data, mean=0.3, std_dev=0.1)
                    noise = self.add_uniform_noise(ori_data)[1]
                    # noise_salt_pepper = ori_data

                    return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(ori_data, 50)[1]).to(device).to(torch.float64), \
                        torch.FloatTensor(noise).to(device).to(torch.float64),
                else:
                    ori_data = self.loader[0][item]
                    # noise = self.add_salt_and_pepper_noise(ori_data, prob=0.2)[1]
                    noise = self.add_uniform_noise(ori_data)[1]
                    # return [torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)]
                    return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(ori_data, 50)[1]).to(device).to(torch.float64), torch.FloatTensor(
                        noise).to(device).to(torch.float64)
            elif self.at == "GSA":
                if self.modelName in self.limitModel:
                    ori_data = self.trainData[item]

                    # noise = self.add_salt_and_pepper_noise(ori_data, prob=0.2)[1]
                    # noise_salt_pepper = self.add_gaussian_noise(ori_data, mean=0.3, std_dev=0.1)
                    noise = self.add_uniform_noise(ori_data)[1]
                    # noise_salt_pepper = ori_data

                    return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(ori_data, 50)[1]).to(device).to(torch.float64), \
                        torch.FloatTensor(noise).to(device).to(torch.float64),
                else:
                    ori_data = self.loader[0][item]
                    # noise = self.add_salt_and_pepper_noise(ori_data, prob=0.2)[1]
                    noise = self.add_uniform_noise(ori_data)[1]
                    # return [torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)]
                    return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(ori_data, 50)[1]).to(device).to(torch.float64), torch.FloatTensor(
                        noise).to(device).to(torch.float64),
            else:

                if self.modelName in self.limitModel:
                    return torch.FloatTensor(self.trainData[item]).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(self.trainData[item], 50)[1]).to(device).to(torch.float64)
                else:
                    # return [torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64)]

                    return torch.FloatTensor(self.loader[0][item]).to(device).to(torch.float64), torch.FloatTensor(
                        self.faster_fourier_transform(self.loader[0][item], 50)[1]).to(device).to(torch.float64)
        else:

            if self.modelName in self.limitModel:
                # return [torch.FloatTensor(self.testData[item]).to(device).to(torch.float64), torch.FloatTensor(
                #     self.labelData[item]).to(
                #     device).to(torch.float64), self.construct_t_adj(self.trainData[item]), self.construct_s_adj(
                #     self.trainData[item].t())]
                ori_data = self.testData[item]
                # noise_salt_pepper = self.add_uniform_noise(ori_data)[0]
                return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                    self.labelData[item]).to(device).to(torch.float64), torch.FloatTensor(
                    self.faster_fourier_transform(self.testData[item], 50)[1]).to(device).to(torch.float64)
            else:
                # return [torch.FloatTensor(self.loader[1][item]).to(device).to(torch.float64), torch.FloatTensor(
                #     self.loader[2][item]).to(
                #     device).to(torch.float64)]
                ori_data = self.loader[1][item]
                # noise_salt_pepper = self.add_uniform_noise(ori_data)[0]
                return torch.FloatTensor(ori_data).to(device).to(torch.float64), torch.FloatTensor(
                    self.loader[2][item]).to(device).to(torch.float64), \
                    torch.FloatTensor(self.faster_fourier_transform(self.loader[1][item], 50)[1]).to(device).to(
                        torch.float64)


def save_model(model, optimizer, scheduler, epoch, accuracy_list, n_windows, item_name, batch=None, desc=""):
    folder = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/'
    print("save model path:", folder)
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims, n_windows, batch=None, item_name="", desc=""):
    import src.models
    model_class = getattr(src.models, modelname)
    if modelname in ["MAC"]:
        model = model_class(dims, n_windows, ablation=batch).double()
    else:
        model = model_class(dims).double()

    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    fname = f'checkpoints_{desc}_{n_windows}/{args.model}_{args.dataset}_{item_name}_{batch}/model.ckpt'
    print("load model path:", fname)
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True, adj_save="", AT="", batchS=0):
    global loss
    feats = dataO
    if model.name == "MAC":
        l = nn.MSELoss(reduction='none')
        l1s = []
        if training:
            if model.name == "MAC":
                if AT == "ADT":
                    l1s = []
                    epoch = []
                    t_adj_q = None
                    s_adj_q = None
                    h = None
                    ori, fd = data
                    for i, d in enumerate(ori):
                        x, t_adj_q, s_adj_q, curv, h, mu, logvar, recon_adj, adj, inter_layer, fusion_inter_layer, noise, gsa = model(
                            d, t_adj_q, s_adj_q, fft=fd[i], hidden=h)
                        KLD = -0.5 / mu.size(0) * torch.mean(
                            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp(), dim=1))
                        BCE = torch.nn.functional.binary_cross_entropy(recon_adj.float(), adj.float())
                        np.save(adj_save[0], t_adj_q.cpu().detach().numpy())
                        np.save(adj_save[1], s_adj_q.cpu().detach().numpy())
                        np.save(adj_save[3], inter_layer)
                        rec_loss = torch.mean(l(x, d))
                        losss = rec_loss + KLD + BCE
                        l1s.append(losss.item())
                        optimizer.zero_grad()
                        losss.backward()
                        optimizer.step()
                elif AT == "Normal":
                    l1s = []
                    epoch = []
                    t_adj_q = None
                    s_adj_q = None
                    h = None
                    ori, fd = data
                    for i, d in enumerate(ori):
                        x, t_adj_q, s_adj_q, curv, h, mu, logvar, recon_adj, adj, inter_layer, fusion_inter_layer, noise, gsa = model(
                            d, t_adj_q, s_adj_q, fft=fd[i], hidden=h)
                        KLD = -0.5 / mu.size(0) * torch.mean(
                            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp(), dim=1))
                        BCE = torch.nn.functional.binary_cross_entropy(recon_adj.float(), adj.float())
                        rec_loss = torch.mean(l(x, d))
                        losss = rec_loss + KLD + BCE
                        l1s.append(losss.item())
                        optimizer.zero_grad()
                        losss.backward()
                        optimizer.step()
                elif AT == "PGD":
                    l1s = []
                    batch_num = 0
                    epoch = []
                    t_adj_q = None
                    s_adj_q = None
                    h = None
                    ori, fd, noise = data
                    eps_iter = 1
                    noise = nn.Parameter(noise[0]).to(device)
                    for i, d in enumerate(ori):
                        x, t_adj_q, s_adj_q, curv, h, mu, logvar, recon_adj, adj, inter_layer, fusion_inter_layer, noise, gsa = model(
                            d, t_adj_q, s_adj_q, fft=fd[i], hidden=h, noise=noise)
                        KLD = -0.5 / mu.size(0) * torch.mean(
                            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp(), dim=1))
                        BCE = torch.nn.functional.binary_cross_entropy(recon_adj.float(), adj.float())
                        rec_loss = torch.mean(l(x, d))
                        losss = rec_loss + KLD + BCE
                        l1s.append(losss.item())
                        optimizer.zero_grad()
                        losss.backward()
                        optimizer.step()
                        delta_new = noise + noise.grad.sign() * eps_iter
                        noise.grad.zero_()
                        delta_new = torch.clip(delta_new, 0., 1.0)
                        delta_new = torch.clip(d + delta_new, 0., 1.0) - d
                        delta_new = delta_new.detach()
                        noise.data.copy_(delta_new)
                elif AT == "GSA":
                    l1s = []
                    batch_num = 0
                    epoch = []
                    t_adj_q = None
                    s_adj_q = None
                    h = None
                    ori, fd, noise = data
                    gsa = torch.bernoulli(torch.full((model.n_feats * 2, model.n_feats * 2), 0.2))
                    for i, d in enumerate(ori):
                        x, t_adj_q, s_adj_q, curv, h, mu, logvar, recon_adj, adj, inter_layer, fusion_inter_layer, noise, gsa = model(
                            d, t_adj_q, s_adj_q, fft=fd[i], hidden=h, noise=None, gsa=gsa)
                        KLD = -0.5 / mu.size(0) * torch.mean(
                            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp(), dim=1))
                        BCE = torch.nn.functional.binary_cross_entropy(recon_adj.float(), adj.float())
                        rec_loss = torch.mean(l(x, d))
                        losss = rec_loss + KLD + BCE

                        l1s.append(losss.item())
                        optimizer.zero_grad()
                        losss.backward()
                        optimizer.step()

                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
                return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            if model.name == "MAC":
                xs = []
                adj_list_t = []
                adj_list_s = []
                jacced = []
                t_adj = torch.tensor(np.load(adj_save[0]), dtype=torch.float64).to(device)
                s_adj = torch.tensor(np.load(adj_save[1]), dtype=torch.float64).to(device)
                if len(data) == 2:
                    ori, fd = data
                else:
                    ori, fd, _ = data
                for i, d in enumerate(ori):
                    x, t, s, _, _, _, _, _, _, _, _, _, _ = model(d.to(device), t_adj, s_adj, fd[i], hidden=None)
                    adj_list_t.append(t)
                    adj_list_s.append(s)
                    coff_t = cacluateJaccard(t_adj=t_adj, test_adj=t, K=model.n_window)
                    coff_s = cacluateJaccard(t_adj=s_adj, test_adj=s, K=model.n_feats)
                    jacced.append(2 * (coff_s * coff_t) / (coff_s + coff_t + 0.00001))
                    xs.append(x)
                adj_list_t = torch.stack(adj_list_t).cpu().detach().numpy()
                adj_list_s = torch.stack(adj_list_s).cpu().detach().numpy()
                xs = torch.stack(xs)
                jacced = torch.stack(jacced)
                y_pred = xs[:, data[0].shape[1] - feats:data[0].shape[1]].view(-1, feats)
                loss = l(xs.to(device), data[0].to(device))
                loss = loss[:, data[0].shape[1] - feats:data[0].shape[1]].view(-1, feats)
                return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), (adj_list_t, adj_list_s, jacced)


def trainModel(model_name, dataset_name, epoch, windows, batch_size, item_dataSet, desc="", AT=""):
    batch_size = batch_size
    model = model_name
    dataset = dataset_name
    args.model = model
    args.dataset = dataset
    new_epoch = epoch
    folder = f'trainRecord/{model_name}_{args.dataset}/{item_dataSet}/{batch_size}/{desc}-{windows}/'
    data_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Train",
                             item_dataSet=item_dataSet, save_path=folder, ATT_mode=AT, )
    features = data_loader.getDim()
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, features, n_windows=windows, desc=desc,
                                                                   batch=batch_size, item_name=item_dataSet)
    model.to(device)
    trainStage = DataLoader(data_loader, batch_size=len(data_loader), shuffle=False)
    test_loader = dataLoader(args.dataset, modelName=model_name, convertWindow=windows, stage="Test", ATT_mode=AT,
                             item_dataSet=item_dataSet, save_path=folder)
    testStage = DataLoader(test_loader, batch_size=len(test_loader), shuffle=False)
    os.makedirs(folder, exist_ok=True)
    train_state_adj = [f'{folder}/{model_name}_t_adj.npy', f'{folder}/{model_name}_s_adj.npy',
                       f'{folder}/{model_name}_curv_{batch_size}.json', f'{folder}/{model_name}_inter_adj.npy', ]

    # train stage
    if not args.test:
        num_epochs = new_epoch
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            model.train()
            train_data = None
            print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
            for train in trainStage:
                train_data = train
                lossT, lr = backprop(e, model, train, features, optimizer, scheduler, training=True, AT=AT,
                                     adj_save=train_state_adj, batchS=batch_size)
                accuracy_list.append((lossT, lr))
                save_model(model, optimizer, scheduler, e, accuracy_list, windows, item_name=item_dataSet,
                           desc=desc,
                           batch=batch_size)
                model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, features, desc=desc,
                                                                               n_windows=windows,
                                                                               batch=batch_size,
                                                                               item_name=item_dataSet)
            torch.zero_grad = True
            model.eval()
            with torch.no_grad():
                print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
                for test, label, fft in testStage:
                    try:
                        labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                        labels_counts = torch.bincount(torch.tensor(labelsFinal))
                    except:
                        labelsFinal = (np.sum(label.cpu().detach().numpy(), axis=1) >= 1) + 0
                        labelsFinal = (np.sum(labelsFinal, axis=1) >= 1) + 0
                        labels_counts = torch.bincount(torch.tensor(labelsFinal))
                    if len(labels_counts) < 2:
                        print(f'{color.HEADER} all is pos samplers {args.dataset}{color.ENDC}')
                        continue

                    if labels_counts[1] < 2:
                        print(f'{color.HEADER} all is neg samplers < 2{args.dataset}{color.ENDC}')
                        continue

                    loss, y_preds, adj = backprop(0, model, (test, fft), features, optimizer, scheduler, training=False,
                                                  adj_save=train_state_adj, batchS=batch_size)

                    lossFinal = np.mean(loss, axis=1)

                    smooth_err = get_err_scores(lossFinal, labelsFinal)
                    rate = 0.1

                    neg_test_index = np.where(labelsFinal > 0)
                    pos_test_index = np.where(labelsFinal == 0)

                    samper_num = int(len(neg_test_index[0]) * rate)
                    if samper_num == 0:
                        samper_num = int(len(neg_test_index[0]) / 2)
                    all_index = set(list(pos_test_index[0])).union(set(list(neg_test_index[0])))
                    sampler_neg = np.random.choice(neg_test_index[0], size=samper_num, replace=False)
                    sampler_pos = np.random.choice(pos_test_index[0], size=samper_num, replace=False)

                    find_best_data = np.concatenate([smooth_err[sampler_neg], smooth_err[sampler_pos]], axis=0)
                    find_best_data_label = np.concatenate([labelsFinal[sampler_neg], labelsFinal[sampler_pos]],
                                                          axis=0)
                    optimal_threshold, _ = search_optimal_threshold(find_best_data, find_best_data_label)

                    overplus_data = smooth_err[list(all_index - set(sampler_neg) - set(sampler_pos))]
                    overplus_data_label = labelsFinal[list(all_index - set(sampler_neg) - set(sampler_pos))]
                    optimal_threshold, optimal_metrics = get_val_res(overplus_data, overplus_data_label,
                                                                     optimal_threshold)
                    print(optimal_metrics)


def getDataSetList(dataSet):
    req = []
    folder = os.path.join(output_folder, dataSet)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    files = glob.glob(os.path.join(folder, "*train.npy"))
    if dataSet == "ASD":
        req = ["omi-10_", ]
    elif dataSet == "MSL":
        req = ["C-1_"]
    elif dataSet == "SMAP":
        req = ["A-4_"]
    elif dataSet == "SMD":
        req = ["machine-3-7_"]
    else:
        for file in files:
            req.append(file[len(folder) + 1:len(file) - len("_train.npy") + 1])
    return req


def getTrainHistory(model, epoch_mod, dataset, batch=None, window="", desc=""):
    history_path = os.getcwd() + "/trainRecord"
    if not os.path.exists(history_path):
        raise Exception('Processed Data not found.')
    need_train = getDataSetList(dataSet=dataset)
    need_train_epoch = [epoch_mod for item in need_train]
    for index, n in enumerate(need_train):
        file_list = glob.glob(os.path.join(history_path, "{}_{}/{}/".format(model, dataset, n)))
        for item in file_list:
            item_path = item + "/{}/{}-{}/model-{}_{}.mat".format(batch, desc, window, model, batch)
            try:
                data = sio.loadmat(item_path)
                epoch = data["epochr"].tolist()[0]
            except:
                epoch = []
            p = epoch_mod - len(epoch)
            if p > 0:
                need_train_epoch[index] = p
            else:
                need_train_epoch[index] = 0
    return need_train_epoch, need_train


if __name__ == '__main__':
    commands = sys.argv[1:]
    Dataset = [args.dataset]
    models = [args.model]
    WindowSize = [args.windowsize]
    epoch = args.epoch
    batch_size = [256]
    AT = [args.attack]
    desc = args.model + "-" + args.space
    for b in batch_size:
        for m in models:
            for d in Dataset:
                for window in WindowSize:
                    need_train_epoch, data_list = getTrainHistory(model=m, epoch_mod=epoch, dataset=d, batch=b,
                                                                  window=window, desc=AT)
                    for e in enumerate(range(epoch)):
                        trainModel(model_name=m, dataset_name=d, epoch=1, windows=window,
                                   batch_size=b, item_dataSet=getDataSetList(dataSet=d)[0], desc=desc, AT=AT)
