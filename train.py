import sys
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from covid19_data import Covid19Dataset
from model import CovidPred

def train_fn(model, train_loader, device, optimizer, criterion, epoch):
    # train
    running_loss = 0.0
    count_iter = 0
    model.train()
    for data in train_loader:
        # get the inputs
        cityData, caseSeries, labels = data
        cityData, caseSeries, labels = cityData.to(device).float(), caseSeries.to(device).float(), labels.to(device).float()
        caseSeries = caseSeries.contiguous().view(*caseSeries.size(), 1)

        if labels.size()[0] == 1:
            continue
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(cityData, caseSeries).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count_iter += 1
    print('Epoch %d loss: %.3f' %
          (epoch + 1, running_loss / count_iter))


def val_fn(model, val_loader, device, criterion, epoch):
    model.eval()
    with torch.no_grad():
        outputs = []
        labels = []
        for data in val_loader:
            cityData, caseSeries, label = data
            cityData, caseSeries, label = cityData.to(device).float(), caseSeries.to(device).float(), label.to(device).float()
            caseSeries = caseSeries.contiguous().view(*caseSeries.size(), 1)
            output = model(cityData, caseSeries)
            outputs.append(output)
            labels.append(label)
        outputs = torch.cat(outputs)
        labels = torch.cat(labels).view(-1)
        loss = criterion(outputs, labels)
        print('Epoch %d val loss: %.3f' % (epoch + 1, loss))
    sys.stdout.flush()

def train_model(batch_size=32, lr=0.0005, regularizer=2E-4):
    train_epochs = 300

    train_set = Covid19Dataset(splits='train')
    val_set = Covid19Dataset(splits='dev')
    model = CovidPred(train_set.city_features(), train_set.sequence_features())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(train_epochs):
        train_fn(model, train_loader, device, optimizer, criterion, i)
        val_fn(model, val_loader, device, criterion, i)

    return model

def evaluate(model, device, city_info, sequence, num_pred = 1):
    model.eval()
    size = sequence.size()[0]
    with torch.no_grad():
        cityData, sequence = city_info.to(device), sequence.to(device)
        for i in range(num_pred):
            output = model(cityData, sequence).contiguous().view(size, 1)
            sequence = torch.cat([sequence, output])
    return sequence


if __name__ == '__main__':
    train_model()