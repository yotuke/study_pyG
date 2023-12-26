import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

dataset = Planetoid(root='TuDatasets', name='Cora')


class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT_Net(dataset.num_node_features, 16, dataset.num_classes, heads=4).to(device)
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
