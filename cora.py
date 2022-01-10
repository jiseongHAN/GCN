import pandas as pd
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from gnn import GCN #nn.Linear
from gcn import GCNN #nn.Parameter

cite = pd.read_table("cora/cora.cites", header=None).sort_values(by=0)
content = pd.read_table("cora/cora.content", header=None).sort_values(by=0)

G = nx.Graph()
for i, j in zip(cite[0], cite[1]):
    G.add_edge(i, j)

A = nx.adjacency_matrix(G).todense()
label = content.iloc[:,-1]
label = np.array(pd.Categorical(label).codes)
one_hot = F.one_hot(torch.LongTensor(label)).float()
feature = np.array(content.iloc[:, 1:-1])

model = GCNN(feature.shape[1], one_hot.shape[1])
train_idx = np.random.choice(A.shape[0], int(A.shape[0] * 0.5), replace=False)
test_idx = np.setdiff1d(np.arange(A.shape[0]), train_idx)

val_list = []
loss_list = []

for i in range(1, 201):
    loss, val_acc = model.train_(A, feature, label, train_idx)
    acc = model.test(A, feature, test_idx, label)
    val_list.append(val_acc)
    loss_list.append(loss)
    print("-" * 30)
    print(f"Epoch {i} Loss: {loss:.2f} Validation ACC: {val_acc * 100:.2f}%")
    print(f"ACC: {acc * 100:.2f}%")
    print("=" * 30)

plt.plot(val_list)
plt.title(label="Validation Accuracy")
plt.savefig("val.png")
plt.show()
plt.plot(loss_list)
plt.title(label="Loss")
plt.savefig("loss.png")
plt.show()


