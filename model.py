import torch
from torch import nn, Tensor
from torch.nn import functional as F

class CovidPred(nn.Module):

    def __init__(self, cityData_features, seq_features):
        super().__init__()
        num_hidden = 32
        self.fc0_1 = nn.Linear(cityData_features, num_hidden * 4)
        self.fc0_2 = nn.Linear(num_hidden * 4, num_hidden * 1)
        self.gru = nn.GRU(seq_features, num_hidden, 1, dropout=0.2)
        self.fc_1 = nn.Linear(num_hidden, seq_features)

    def forward(self, cityData:Tensor, caseSeries:Tensor):
        cityData = torch.zeros_like(cityData)
        caseSeries = caseSeries.permute(1, 0, 2).log1p()
        h = F.leaky_relu(self.fc0_1(cityData), 0.1)
        h = F.leaky_relu(self.fc0_2(h), 0.1).contiguous().view(cityData.size()[0], self.gru.num_layers, self.gru.hidden_size).permute(1, 0, 2)
        output, h = self.gru(caseSeries, h)
        h = h[-1]
        return self.fc_1(torch.cat([h], dim=1)).exp() - 1
