import sys
sys.path.append('/opt/anaconda3/envs/pytorch/lib/python3.9/site-packages')

print("Python executable:", sys.executable)
print("Python path:", sys.path)

try:
    import scanpy
    print("Scanpy imported successfully!")
except ModuleNotFoundError as e:
    print("Scanpy import failed:", e)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphConv
from torch_geometric.utils import add_self_loops, degree


class InnerProductDecoder(nn.Module):
    """
    使用节点嵌入的内积作为重构的边权预测。
    """
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.t())
        return self.activation(adj)
class GCNAE(nn.Module):
    """
    使用 PyGWeightedConv 实现的 Graph Conv AE (自编码器).
    支持多层 Graph Conv 层。
    """
    def __init__(self, in_feats, n_hidden, n_layers=1,
                 activation=None, dropout=0.1,
                 hidden=None, hidden_relu=False, hidden_bn=False):
        """
        Args:
            in_feats (int): 输入特征维度
            n_hidden (int): 隐藏层特征维度
            n_layers (int): 层数
            activation (callable): 激活函数
            dropout (float): dropout 比例
            hidden (list or None): 用于自定义隐藏层的特征维度
            hidden_relu (bool): 是否对隐藏层启用 ReLU 激活
            hidden_bn (bool): 是否对隐藏层使用 BatchNorm
        """
        super(GCNAE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.hidden_relu = hidden_relu
        self.hidden_bn = hidden_bn
        self.batchnorms = nn.ModuleList()

        # 添加 GraphConv 层
        for i in range(n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            self.layers.append(GraphConv(in_dim, n_hidden))

            # 如果需要 BatchNorm
            if self.hidden_bn:
                self.batchnorms.append(nn.BatchNorm1d(n_hidden))

        # 添加解码器 (InnerProductDecoder)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。

        Args:
            x (torch.Tensor): 节点特征
            edge_index (torch.Tensor): 图的边信息
        
        Returns:
            adj_logits (torch.Tensor): 重构的邻接矩阵 logits
            x (torch.Tensor): 节点嵌入
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)

            if self.hidden_bn:
                x = self.batchnorms[i](x)

            if self.activation:
                x = self.activation(x)

            if self.dropout:
                x = self.dropout(x)

        # 使用解码器重构邻接矩阵
        adj_logits = self.decoder(x)
        return adj_logits, x


# 调试代码
if __name__ == "__main__":
    print("Testing GCNAE model...")

    # 模拟输入
    num_nodes = 100
    in_feats = 50
    n_hidden = 128
    n_layers = 2

    x = torch.randn((num_nodes, in_feats))
    edge_index = torch.randint(0, num_nodes, (2, 300))

    # 初始化模型
    model = GCNAE(in_feats=in_feats, n_hidden=n_hidden, n_layers=n_layers,
                  activation=F.relu, dropout=0.1, hidden_bn=True)
    print(model)

    # 检查模型参数
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Shape: {param.shape}")

    # 测试前向传播
    adj_logits, node_embeddings = model(x, edge_index)
    print(f"Adjacency logits shape: {adj_logits.shape}")
    print(f"Node embeddings shape: {node_embeddings.shape}")