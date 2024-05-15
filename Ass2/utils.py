import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections.abc import Sequence
from math import ceil


fillna = dict( # Different values to replace NaNs in each column
    visitor_hist_starrating = -1,
    visitor_hist_adr_usd = -1,
    prop_review_score = -1,
    prop_location_score2 = -1,
    srch_query_affinity_score = -1,
    orig_destination_distance = -1,
    **{f"comp{i}_rate": 0 for i in range(1, 9)},
    **{f"comp{i}_inv": -1 for i in range(1, 9)},
    **{f"comp{i}_rate_percent_diff": -1 for i in range(1, 9)}
)
missing = {f"comp{i}_rate_missing": f"comp{i}_rate" for i in range(1, 9)}
pca = pickle.load(open("models/pca.pkl", 'rb'))
cols = open('data/cols.txt', 'r').read().split('\n')


class Identity(nn.Module):
    def forward(self, x):
        return x

class DataLoader(Sequence):
    def __init__(self, data, bs):
        self.data = data
        self.bs = bs
        
    def __len__(self):
        return ceil(len(self.data) / self.bs)
    
    def __getitem__(self, i):
        return self.data[i*self.bs:(i+1)*self.bs if i != -1 else None]
    
    def __iter__(self):
        self.i = -1
        return self
    
    def __next__(self):
        self.i += 1
        if self.i >= len(self): raise StopIteration
        return self[self.i]
    
    def shuffle(self):
        self.data = self.data.sample(frac=1)
    

# Add additional columns where filling in NaNs with special values isn't enough
def fillnans(df):
    for k, v in missing.items():
        df[k] = df[v].isna().astype(int)
    return df.fillna(fillna)

def split_time(df):
    # Split date_time into year, month, day, and time
    dates = np.vstack(df.date_time.dt.strftime("%Y,%m,%d,%H,%M,%S").str.split(',').map(np.asarray)).astype(int)
    dates[:,0] -= 2000 # Years before 2000 are negative
    df[['year', 'month', 'day']] = dates[:,:3]
    df['time'] = (dates[:,3] * 3600 + dates[:,4] * 60 + dates[:,5]) / 86400 # Time in range 0 to 1
    return df.drop(columns='date_time')

def clean(df):
    df = df[~(df.price_usd > df.price_usd.mean() + 3 * df.price_usd.std())] # Drop outliers
    return df[df.price_usd != 0] # Drop instances where the property costs nothing

# One-hot encode categorical values
def encode(df, enc):
    # Reduce the size of the one-hot encoded variables with PCA
    return pd.concat([df, pd.DataFrame(pca.transform(enc[df.index]), index=df.index)], axis=1)

# TOO EXPENSIVE
# USE PRECOMPUTED VALUES INSTEAD
# # One-hot encode categorical values
# def encode(df):
#     df = pd.get_dummies(df, columns=['site_id', 'visitor_location_country_id', 'prop_country_id'], dtype=bool)
#     df.loc[:,list(set(cols).difference(set(df.columns)))] = False
#     df = df.drop(columns=list(set(df.columns[-len(cols):]).difference(set(cols))))
#     # Reduce the size of the one-hot encoded variables with PCA
#     return pd.concat([df.iloc[:,:-len(cols)], pd.DataFrame(pca.transform(df[cols]), index=df.index)], axis=1)

def split_data_and_targets(df, regression=True):
    cols = ['prop_id', 'srch_id', 'position', 'gross_bookings_usd', 'click_bool', 'booking_bool']
    try: 
        metadata = df[cols]
        if regression:
            metadata.loc[:, 'booking_bool'] *= 5
            metadata.loc[:, 'target'] = metadata.iloc[:,-2:].max(axis=1)
        else:
            metadata.loc[:, 'booking_bool'] *= 2
            metadata.loc[:, 'target'] = metadata.iloc[:,-2:].max(axis=1).astype(int)
    except KeyError:
        cols = ['prop_id', 'srch_id']
        metadata = df[cols]
    return df.drop(columns=cols[1:]), metadata

# def pipeline(df, ret='torch', regression=True):
#     df = encode(df)
#     df, metadata = split_data_and_targets(df, regression=regression)
#     if ret == 'pandas':
#         return df, metadata
#     if ret == 'numpy':
#         return df.to_numpy(), metadata.target.to_numpy()
#     x = torch.tensor(df.to_numpy(), dtype=torch.float32, device='cuda')
#     y = torch.tensor(metadata.target.to_numpy(), dtype=torch.float32, device='cuda').reshape(-1, 1)\
#         if regression else torch.tensor(metadata.target.to_numpy(), dtype=torch.int64, device='cuda')
#     return x, y

# def train(model, opt, criterion, data, r=True):
#     losses = list()
#     for x in tqdm(data):
#         x, y = pipeline(x, regression=r)
#         opt.zero_grad()
#         loss = criterion(model(x), y)
#         loss.backward()
#         opt.step()
#         losses.append(loss.detach().cpu())
#     return losses

def pipeline(df, enc, ret='torch', regression=True):
    df = encode(df, enc)
    df, metadata = split_data_and_targets(df, regression=regression)
    if ret == 'pandas':
        return df, metadata
    if ret == 'numpy':
        return df.to_numpy(), metadata.target.to_numpy()
    x = torch.tensor(df.to_numpy(), dtype=torch.float32, device='cuda')
    y = torch.tensor(metadata.target.to_numpy(), dtype=torch.float32, device='cuda').reshape(-1, 1)\
        if regression else torch.tensor(metadata.target.to_numpy(), dtype=torch.int64, device='cuda')
    return x, y

def train(model, opt, criterion, data, enc, r=True):
    losses = list()
    for x in tqdm(data):
        x, y = pipeline(x, enc, regression=r)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach().cpu())
    return losses

def plot_loss(loss, model, save=False):
    plt.plot(loss)
    plt.xlabel("Batch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"Loss curve for {model} model", fontsize=18)
    if bool(save): plt.savefig(f"images/{save}.png")