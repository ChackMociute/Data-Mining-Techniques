import pickle
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *

warnings.filterwarnings("ignore", message="X does not have valid feature names, but IncrementalPCA was fitted with feature names")
pd.options.mode.chained_assignment = None


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.mlp_r = self.load_MLP('models/MLP_reg.pt')
        self.mlp_c = self.load_MLP('models/MLP_class.pt')
        # self.rf_models = [pickle.load(open(f'models/rf_{m}.pkl', 'rb'))
        #                   for m in ['book', 'click', 'large', 'small']]
        # for rf in self.rf_models:
        #     rf.n_jobs = 1
        self.seq1 = nn.Sequential(
            nn.BatchNorm1d(220),
            nn.Linear(220, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(192, 512), # 216
            self.relu,
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            self.relu,
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            self.relu,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(192, 512), # 216
            self.relu,
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            self.relu,
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            self.relu,
            nn.Linear(64, 1)
        )
    
    def load_MLP(self, path):
        model = torch.load(path)
        for i in ['19', '20', '21']:
            model.__setattr__(i, Identity())
        self.fine_tune_params = []
        for p in model.parameters():
            if p.requires_grad:
                self.fine_tune_params.extend([p])
            p.requires_grad = False
        return model
    
    def get_fine_tune_params(self):
        for p in self.fine_tune_params:
            p.requires_grad = True
        return p
        
    
    def forward(self, x):
        bs = len(x)
        x = x.reshape(2 * bs, -1)
        # y = np.asarray([m.predict_proba(x) for m in self.rf_models])
        # y = torch.tensor(y.reshape(bs, 2, -1), dtype=torch.float32, device='cuda')
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
        x = torch.stack([self.lin1(self.mlp_c(x)), self.lin2(self.mlp_r(x)), self.seq1(x)], dim=1)
        x = self.flatten(self.relu(x).reshape(bs, 2, -1))
        # x = torch.cat([self.flatten(x), self.flatten(y)], dim=1)
        p = self.seq2(x).squeeze()
        m = torch.exp(self.seq3(x)).squeeze()
        return p, torch.sign(p-0.5) * m

if __name__ == "__main__":
    print("Loading data")
    full = pd.read_csv('data/training_set_VU_DM.csv', parse_dates=['date_time'])
    print("Loaded training csv file")
    full = full.drop(columns=['site_id', 'visitor_location_country_id', 'prop_country_id'])
    full = clean(split_time(fillnans(full))).reset_index(drop=True)
    print("Finished preprocessing")
    enc = np.load('data/enc_train.npy')
    print("Loaded encodings")
    data = DataLoader(pd.read_csv('data/targets.csv', index_col=False), 8192)
    print("Loaded targets")
    
    
    r = Ranker().to('cuda')
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': [p for p in r.parameters() if p.requires_grad]},
                                  {'params': r.get_fine_tune_params(), 'lr': 2e-4}],
                                 lr=3e-3, weight_decay=1e-4)
    
    print("Starting training")
    losses1 = list()
    losses2 = list()
    for e in range(5):
        r.train()
        data.shuffle()
        loss1 = list()
        loss2 = list()
        for i, x in enumerate(tqdm(data)):
            optimizer.zero_grad()

            # Generate data and labels
            y1 = torch.tensor(x.greater.to_numpy(), dtype=torch.float32, device='cuda')
            y2 = full.loc[x[['first', 'second']].values.flatten()].position.to_numpy().reshape(-1, 2)
            y2 = torch.tensor(y2[:,0] - y2[:,1], dtype=torch.float32, device='cuda')
            x = pipeline(full.loc[x[['first', 'second']].values.flatten()], enc, ret='numpy')[0].reshape(-1, 2, 220)

            # Get loss and update model
            p, m = r(x)
            l1 = criterion1(p, y1)
            l2 = criterion2(m, y2)
            loss = l1 + l2 / 200
            loss.backward()
            optimizer.step()

            loss1.append(l1.detach().cpu())
            loss2.append(l2.detach().cpu())
            # if i % 30 == 0:
            #     losses1.extend(loss1)
            #     print(f"l1: {sum(loss1)/len(loss1)}")
            #     loss1 = list()
            #     losses2.extend(loss2)
            #     print(f"l2: {sum(loss2)/len(loss2)}")
            #     loss2 = list()
        losses1.extend(loss1)
        losses2.extend(loss2)
        print(f"Epoch {e} l1: {sum(loss1)/len(loss1)}")
        print(f"Epoch {e} l2: {sum(loss2)/len(loss2)}")
    
    torch.save(r, 'ranker.pt')
    torch.save({
        'model': r.state_dict(),
        'optimizer': optimizer.state_dict(),
        'l1': losses1,
        'l2': losses2
    }, 'ranker_checkpint.pt')
    
    plot_loss(losses, 'Ranker')
    plt.show()