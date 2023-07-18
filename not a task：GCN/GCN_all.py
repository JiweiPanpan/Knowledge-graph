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
heads = set()
tails = set()
relations = set()

# 收集实体和关系
for triple in dataset:
    head, relation, tail = triple
    heads.add(head)
    tails.add(tail)
    relations.add(relation)

# 构建实体和关系的映射字典
head2id = {head: idx for idx, head in enumerate(heads)}
tail2id = {tail: idx for idx, tail in enumerate(tails)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}


# 构建节点特征和边索引
edge_index = []
head_features = []
tail_features = []
relation_features= []
for triple in dataset:
    head, relation, tail = triple
    head_idx = head2id[head]
    tail_idx = tail2id[tail]
    relation_idx = relation2id[relation]

    # 添加节点特征
    head = get_word_embeddings(head)
    tail = get_word_embeddings(tail)
    relation= get_word_embeddings(relation)

    head = torch.mean(head, dim=1)
    tail = torch.mean(tail, dim=1)
    relation = torch.mean(relation, dim=1)


    # x.append(torch.cat([head, tail], dim=0))
    head_features.append(head)
    tail_features.append(tail)
    relation_features.append(relation)
    # 添加边索引
    edge_index.append([head_idx, tail_idx])
    # edge_index.append([tail_idx, head_idx])
    # 添加节点标签


# 转换为PyG的Data对象
Head = torch.stack(head_features, dim=1).squeeze(0)
Tail = torch.stack(tail_features, dim=1).squeeze(0)
Relation = torch.stack(relation_features, dim=1).squeeze(0)

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 构建邻接矩阵
num_nodes = Head.size(0)
adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
for edge in edge_index.t():
    src, tgt = edge.tolist()

    adjacency_matrix[src, tgt] = 1.0

# data = Data(Head=Head, Tail=Tail,edge_index=edge_index, adj=adjacency_matrix, train_mask=train_mask,  test_mask=test_mask)
data = Data(Head=Head, Tail=Tail,edge_index=edge_index)

# 步骤4：构建GCN模型



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

class HeadModel(torch.nn.Module):
    def __init__(self, gcn_model):
        super(HeadModel, self).__init__()
        self.gcn_model = gcn_model

    def forward(self, x, edge_index):
        gcn_output = self.gcn_model(x, edge_index)


        return gcn_output


class TailModel(torch.nn.Module):
    def __init__(self, gcn_model):
        super(TailModel, self).__init__()
        self.gcn_model = gcn_model

    def forward(self, x, edge_index):
        gcn_output = self.gcn_model(x, edge_index)


        return gcn_output

class RelationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RelationModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.classifier = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, Relation):
        Relation = self.classifier(Relation)
        return Relation


relation_model = RelationModel(Relation.size(1), 32)
Relation = relation_model(Relation)
print(f'RELATION: {Relation.size()}')


input_dim = Head.size(1)
hidden_dim = 64
output_dim = 32

gcn_model = GCN(input_dim, hidden_dim, output_dim)

# Create the relation prediction model
num_relations = len(relations)
head_model = HeadModel(gcn_model)
tail_model = TailModel(gcn_model)

# Set the model in training mode
head_model.train()
tail_model.train()

# Define the loss function and optimizer

optimizer_head = torch.optim.Adam(head_model.parameters(), lr=0.01,)
optimizer_tail = torch.optim.Adam(tail_model.parameters(), lr=0.01,)




gcn_output_head = head_model(Head, edge_index)
gcn_output_tail = tail_model(Tail, edge_index)

# split
num_samples = gcn_output_head.size(0)
split_index = int(0.8 * num_samples)
train_head = gcn_output_head[:split_index]
test_head = gcn_output_head[split_index:]
train_tail = gcn_output_tail[:split_index]
test_tail = gcn_output_tail[split_index:]
train_relation = Relation[:split_index]
test_relation = Relation[split_index:]




print(f'train_head: {train_head}')
print(f'train_tail: {train_tail}')
print(f'train_head size: {train_head.size()}')
print(f'train_tail size: {train_tail.size()}')
print(f'train_relation size: {train_relation.size()}')



print(f'gcn_output_head: {gcn_output_head}')
print(f'gcn_output_head size: {gcn_output_head.size()}')
print(f'gcn_output_tail: {gcn_output_tail}')
print(f'gcn_output_tail size: {gcn_output_tail.size()}')








