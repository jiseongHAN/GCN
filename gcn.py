import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GCNN(nn.Module):
    def __init__(self, n_input, n_output, h_weight=32):
        super(GCNN, self).__init__()

        self.W1 = nn.Parameter(torch.randn(n_input, h_weight), requires_grad=True)
        self.W1_bias = nn.Parameter(torch.randn(h_weight), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(h_weight, n_output), requires_grad=True)
        self.W2_bias = nn.Parameter(torch.randn(n_output), requires_grad=True)

        nn.init.xavier_uniform_(self.W1)

        nn.init.xavier_uniform_(self.W2)


        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def pre_processing(self, matrix):
        mat_tilde = np.identity(matrix.shape[0]) + matrix
        D = np.diag(np.asarray(1 / np.sum(mat_tilde, axis=1)).squeeze())
        # D^(1/2) is just sqrt of each diagonal element cause D is diagonal matrix
        a_hat =np.matmul(np.matmul(np.sqrt(D), mat_tilde), np.sqrt(D))
        return torch.FloatTensor(a_hat)

    def normalize(self, x):
        return (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)

    def forward(self, A, feature):
        if not isinstance(feature, torch.Tensor):
            feature = torch.FloatTensor(feature)

        feature = self.normalize(feature)
        a_hat = self.pre_processing(A)

        ret = torch.mm(feature, self.W1) + self.W1_bias
        ret = torch.mm(a_hat, ret)
        ret = F.relu(ret)

        ret = F.dropout(ret, p=0.5, training=self.training)
        ret = torch.mm(ret, self.W2) + self.W2_bias
        ret = torch.mm(a_hat, ret)

        return F.log_softmax(ret, 1)

    def train_(self, A, feature, answer, idx):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(A, feature)
        pred = out[idx]
        loss = F.nll_loss(pred, torch.LongTensor(answer[idx]))
        loss.backward()
        self.optimizer.step()

        acc = (out[idx].argmax(-1).numpy() == answer[idx]).mean()

        return loss.item(), acc

    def test(self, A, feature, idx, label):
        self.eval()
        output = self.forward(A, feature)
        acc = (output[idx].argmax(-1).numpy() == label[idx]).mean()
        return acc

if __name__=="__main__":
    example = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    feature = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    answer = torch.LongTensor([0, 0, 0, 0])

    net = GCNN(feature.shape[1], answer.shape[1])
    for i in range(3):
        loss, acc = net.train(example, feature, answer)
        print("-" * 30)
        print(f"Epoch {i} Loss: {loss} Validation ACC : {acc}")
        print(f"Output: {net.forward(example, feature)}")
        print("=" * 30)