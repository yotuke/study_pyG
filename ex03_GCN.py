
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='TuDatasets', name='Cora')


class GCN_Net(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(feature, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)   # edge_index 会自动被求出邻接矩阵
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam([
# 	dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
#     ], lr=0.01)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    correct = out[data.train_mask].max(dim=1)[1].eq(data.y[data.train_mask]).double().sum()
    # print('epoch:', epoch, ' acc:', correct / int(data.train_mask.sum()))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 9:
        model.eval()
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        log = 'Epoch: {:03d}, Train: {:.5f}, Val: {:.5f}, Test: {:.5f}'
        print(log.format(epoch + 1, accs[0], accs[1], accs[2]))

model.eval()   
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(acc)
