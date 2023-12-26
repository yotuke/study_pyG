"""
pyG入门：
    1.单个图是 torch_geometric.data.Data()类的一个实例
        data = torch_geometric.data.Data(x， edge_index)
            x: 节点的特征矩阵，形如 [num_nodes, num_node_features]
                num_nodes: 节点个数
                num_node_features: 每个节点的特征数
            edge_index: 边的索引，形如[2, num_edges]
                2: 表示边的起点和终点存储在两个列表中，第一个列表存储起点，第二个列表存储终点
                num_edges: 边的个数
    2.pyG的输入都是tensor类型的
    3.用pyG表示无向图时，一条边要统计两遍，代表两条不同的起点和终点
    4.常用命令
        print(data.keys)
        print(data['x'])                # 查看节点的属性信息
        print(data.num_nodes)           # 节点数
        print(data.num_edges)           # 边数
        print(data.num_node_features)   # 节点属性向量的维度
        print(data.has_isolated_nodes())# 图中是否有孤立节点
        print(data.has_self_loops())    # 图中是否有环
        print(data.is_directed())       # 是否是有向图
"""
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [2], [3]], dtype=torch.float)

data = Data(x, edge_index)
print(data)
print(data.keys)    # ['x', 'edge_index']
print(data["x"])
print(data["edge_index"])
print(data.edge_index)