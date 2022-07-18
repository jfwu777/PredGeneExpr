import numpy as np

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from tqdm import tqdm


class NaiveLSTM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size=6
                 ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.lstm(x)
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)
        h_concat = torch.cat((avg_pool, max_pool), 1)

        x = self.mlp(h_concat)
        return x


def data2dl_cv(data_file, batch_size, split=0.8):
    train_data = torch.load(data_file)

    sequence, score = train_data
    idx = np.arange(len(sequence))
    np.random.shuffle(idx)
    split_point = int(len(sequence)*0.8)
    train_sequence = sequence[:split_point]
    test_sequence = sequence[split_point:]
    train_score = score[:split_point]
    test_score = score[split_point:]

    train_dataset = TensorDataset(train_sequence, train_score)
    test_dataset = TensorDataset(test_sequence, test_score)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def data2dl(data_file, batch_size):
    train_data = torch.load(data_file)

    sequence, score = train_data
    dataset = TensorDataset(sequence, score)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def main():
    # args
    max_epoch = 100
    batch_size = 4
    train_data_file = 'train_mini.pt'
    test_data_file = 'train_mini.pt'
    logfile = 'log/naive_lstm_mini.log'
    device = 0

    fp = open(logfile, 'w')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:%s' % device)
    else:
        device = torch.device('cpu')

    trainloader, testloader = data2dl_cv(train_data_file, batch_size)
    # testloader = data2dl(test_data_file, batch_size)

    model = NaiveLSTM(
        embedding_dim=128,
        hidden_dim=64,
        vocab_size=6, # A,G,C,T + N + <PAD>
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(max_epoch):
        # train
        model.train()
        epoch_loss = 0
        for i, data in tqdm(enumerate(trainloader)):
            seq, score = data
            seq = seq.to(device)
            score = score.to(device)

            optimizer.zero_grad()
            pred = model(seq)
            score = score.type(torch.float)
            loss = criterion(pred.squeeze(), score)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #eval
        model.eval()
        preds = []
        scores = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader)):
                seq, score = data
                seq = seq.to(device)
                score = score.to(device)
                pred = model(seq)
                preds.append(pred.cpu().numpy().squeeze())
                scores.append(score.cpu().numpy())

        preds = np.concatenate(preds)
        scores = np.concatenate(scores)
        pearsonr, _ = stats.pearsonr(preds, scores)
        spearmanr, _ = stats.spearmanr(preds, scores)
        logtxt = f'epoch {epoch} | pearsonr {pearsonr:.2f} | spearmanr {spearmanr:.2f} | loss {epoch_loss:.2f}'
        fp.writelines(logtxt + '\n')
        print(logtxt)

    fp.close()


if __name__ == '__main__':
    main()