"""
PyG包含了一些常用的图深度学习公共数据集，可以直接导入使用如
    Planetoid数据集（Cora、Citeseer、Pubmed）
    一些来自于http://graphkernels.cs.tu-dortmund.de常用的图神经网络分类数据集
    QM7、QM9
    3D点云数据集，如FAUST、ModelNet10等
"""
from torch_geometric.datasets import TUDataset, Planetoid

dataset = Planetoid('TuDatasets', 'Cora')    # 下载数据集,dataset是个图列表 dataset=[g1, g2, ..., gn]

print("数据集中图的个数 ：{}".format(len(dataset)))          # len(dataset) 数据集中图的个数
print("dataset.num_classes = {}".format(dataset.num_classes))
print("dataset.num_node_features = {}".format(dataset.num_node_features))
for graph in dataset:
    print(graph)
    print(graph.num_nodes)

