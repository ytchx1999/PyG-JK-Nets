from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import JumpingKnowledge

dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./cora/', name='Cora', split='random',
#                          num_train_per_class=232, num_val=542, num_test=542)
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/', name='Pubmed')
print(dataset)


# baseline：GCN模型（2层）
class GCNNet(nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


# baseline：GAT模型（2层）
class GATNet(nn.Module):
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# JK-Nets（6层）
class JKNet(nn.Module):
    def __init__(self, dataset, mode='max', num_layers=6, hidden=16):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode

        self.conv0 = GCNConv(dataset.num_node_features, hidden)
        self.dropout0 = nn.Dropout(p=0.5)

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), GCNConv(hidden, hidden))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, dataset.num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(F.relu(conv(x, edge_index)))
            layer_out.append(x)

        h = self.jk(layer_out)  # JK层

        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h


model = JKNet(dataset, mode='max')  # max和cat两种模式可供选择
# model = GCNNet(dataset)
# model = GATNet(dataset)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 按照60%、20%、20%划分train、valid、test
if dataset.name == 'Cora':
    data.train_mask[:1624] = True
    data.train_mask[1624:2166] = True
    data.train_mask[2166:] = True
elif dataset.name == 'Citeseer':
    data.train_mask[:1995] = True
    data.train_mask[1995:2661] = True
    data.train_mask[2661:] = True
elif dataset.name == 'Pubmed':
    data.train_mask[:11829] = True
    data.train_mask[11829:15773] = True
    data.train_mask[15773:] = True


def train():
    model.train()
    for epoch in range(100):
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out[data.train_mask], dim=1)
        correct = (pred == data.y[data.train_mask]).sum().item()
        acc = correct / data.train_mask.sum().item()

        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
            epoch, loss.item(), acc))

        # val_loss, val_acc = valid()

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))

    test()


# def valid():
#     # model.eval()
#     with torch.no_grad():
#         out = model(data)
#         loss = criterion(out[data.val_mask], data.y[data.val_mask])
#         _, pred = torch.max(out[data.val_mask], dim=1)
#         correct = (pred == data.y[data.val_mask]).sum().item()
#         acc = correct / data.val_mask.sum().item()
#         return loss.item(), acc
#         # print("val_loss: {:.4f} val_acc: {:.4f}".format(loss.item(), acc))


def test():
    model.eval()
    out = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))


if __name__ == '__main__':
    train()
