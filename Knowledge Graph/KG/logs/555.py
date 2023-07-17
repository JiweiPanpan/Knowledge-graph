import torch
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as f

# 步骤1：读取数据集文本文件
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    dataset = []
    for line in lines:
        triple = line.strip().split('\t')
        dataset.append(triple)
    return dataset


dataset_file = 'C:/Users/panji/OneDrive/Desktop/Ourmodel/3-TransH/datasets/new dataset/train.txt'  # 数据集文件路径
dataset = read_dataset(dataset_file)

# 步骤2：使用BERT进行单词嵌入
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def get_word_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = bert_model(**inputs)
    word_embeddings = outputs.last_hidden_state
    return word_embeddings


# 步骤3：构建知识图谱
entities = set()
relations = set()

# 收集实体和关系
for triple in dataset:
    head, relation, tail = triple
    entities.add(head)
    entities.add(tail)
    relations.add(relation)

# 构建实体和关系的映射字典
entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}


# 构建节点特征和边索引
x = []
edge_index = []
node_features = []
y = []
for triple in dataset:
    head, relation, tail = triple
    head_idx = entity2id[head]
    tail_idx = entity2id[tail]
    relation_idx = relation2id[relation]

    # 添加节点特征
    head = get_word_embeddings(head)
    tail = get_word_embeddings(tail)
    head = torch.mean(head, dim=1)
    tail = torch.mean(tail, dim=1)
    # x.append(torch.cat([head, tail], dim=0))
    node_features.append(head)
    node_features.append(tail)
    # 添加边索引
    edge_index.append([head_idx, tail_idx])
    edge_index.append([tail_idx, head_idx])
    # 添加节点标签

    y.append(relation_idx)

# 转换为PyG的Data对象
x = torch.stack(node_features, dim=1).squeeze(0)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
y = torch.tensor(y, dtype=torch.long).unsqueeze(1)
# 构建邻接矩阵
num_nodes = x.size(1)
adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
for edge in edge_index.t():
    src, tgt = edge.tolist()

    adjacency_matrix[src, tgt] = 1.0

# 创建 train_mask

train_indices = list(range(0, int(0.8 * num_nodes)))  # 80% 的数据作为训练集
test_indices = list(range(int(0.8 * num_nodes), num_nodes))  # 20% 的数据作为测试集

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True  # 将训练集样本的索引设置为 True

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_indices] = True  # 将测试集样本的索引设置为 True

data = Data(x=x, edge_index=edge_index, adj=adjacency_matrix, y=y, train_mask=train_mask,  test_mask=test_mask)

# 步骤4：构建GCN模型

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#
#         self.classifier = nn.Linear(output_dim, 1)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()  # Final GNN embedding space.
#
#         # Apply a final (linear) classifier.
#         out = self.classifier(h)
#
#         return out

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = f.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = f.relu(x)
        x = self.conv3(x, edge_index)

        return x
# 步骤5：创建并使用GCN模型

input_dim = x.size(1)
hidden_dim = 64
output_dim = 32

# model = GCN(input_dim, hidden_dim, output_dim)
# outputs = model(data.x, data.edge_index)
#
# print(outputs)

model = GCN(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = f.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
#%%
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')