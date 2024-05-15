import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

origin_net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

origin_net_data = torch.load("lenet_origin.pth")
origin_net.load_state_dict(origin_net_data.state_dict())

output_dim = 5

"""
    Graph Convolution Network
"""

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, input_dim=25, output_dim=1):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 6)
        self.gcn2 = GraphConvolution(6, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, adjacency, conv1, conv2, i):
        kernel_size_height = conv1.weight.shape[2]
        kernel_size_width = conv1.weight.shape[3]
        x1 = torch.reshape(conv1.weight, (-1, kernel_size_height * kernel_size_width))
        x2 = torch.reshape(conv2.weight, (-1, kernel_size_height * kernel_size_width))
        feature = torch.cat((x1, x2), dim=0)

        """Uniform distribution noise"""
        junyun= torch.rand(feature.shape) * 2 - 1
        junyun = junyun * i
        feature = feature + junyun

        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        logits = self.sig(logits)
        return logits


tensor_adjacency = torch.load("lenet_adjacency_480bits_star_topology.pkl")
tensor_adjacency = tensor_adjacency.to("cpu")

gcn_model = torch.load("lenet_gcn_480bits_star_topology.pth")
gcn_model = gcn_model.to("cpu")

"""
    decision
"""

def decision(secrets_hat):
    with torch.no_grad():
        length0 = secrets_hat.shape[0]
        length1 = secrets_hat.shape[1]
        for i in range(length0):
            for j in range(length1):
                if secrets_hat[i][j] > 0.5:
                    secrets_hat[i][j] = 1
                else:
                    secrets_hat[i][j] = 0
        decision_hat = secrets_hat
    return decision_hat

"""
    accuracy
"""

def gcn_accuracy(Y_decision, Y):
    with torch.no_grad():
        sum = Y_decision.numel()
        correct = 0
        error = 0
        length0 = Y_decision.shape[0]
        length1 = Y_decision.shape[1]
        for i in range(length0):
            for j in range(length1):
                if Y_decision[i][j] == Y[i][j]:
                    correct += 1
                else:
                    error += 1
        correct_rate = correct/sum
        error_rate = error/sum
        # print("correct rate：", correct_rate)
        # print("error rate：", error_rate)
    return correct_rate, error_rate

"""
    ascii binarys to string
"""

def ascii2str(result_binarys):
    all_binarys = []
    result_str = ""
    for binarys in result_binarys:
        all_binarys += binarys
    count = 0
    temp_str = ""
    for bit in all_binarys:
        temp_str += str(bit)
        count += 1
        if count % 8 == 0:
            result_str += chr(int(temp_str, 2))
            temp_str = ""
    return result_str

"""
    about KL Divergence
"""

class Frequency:
    def __init__(self, n = 800):
        self.n = n
        self.data = [0.0] * n
        self.center = round(n / 2)
        self.space = 32 / n
        self.x = []
        for i in range(n):
            self.x.append( (i-self.center) * self.space)

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def frequency(self, tensor):
        shape = tensor.shape
        size = len(shape)
        if size == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    index = torch.round(tensor[i][j] / self.space) + self.center
                    # print(index)
                    if index > (self.n - 1):
                        index = torch.tensor(self.n - 1)
                    if index < 0:
                        index = torch.tensor(0)
                    self.data[int(index.item())] += 1

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def calculate_KL_divergence(origin_net, lenet):
    kernel_size_height = origin_net[0].weight.shape[2]
    kernel_size_width = origin_net[0].weight.shape[3]
    P1 = torch.reshape(origin_net[0].weight, (-1, kernel_size_height * kernel_size_width))
    P2 = torch.reshape(origin_net[3].weight, (-1, kernel_size_height * kernel_size_width))
    Q1 = torch.reshape(lenet[0].weight, (-1, kernel_size_height * kernel_size_width))
    Q2 = torch.reshape(lenet[3].weight, (-1, kernel_size_height * kernel_size_width))
    origin_frequency = Frequency()
    stego_frequency = Frequency()
    origin_frequency.frequency(P1)
    origin_frequency.frequency(P2)
    stego_frequency.frequency(Q1)
    stego_frequency.frequency(Q2)
    p = torch.tensor(origin_frequency.data)
    q = torch.tensor(stego_frequency.data)
    for i in range(len(q)):
        if p[i] != 0 and q[i] == 0:
            q[i] = 0.0001
    p = p / p.sum()
    q = q / q.sum()
    divergence = F.kl_div(input=q.log(), target=p, reduction='sum')
    return divergence

for i in range(20):
    secret_tensor = torch.load("lenet_secret_480bits"+str(i)+".pkl")
    secret_tensor = secret_tensor.to("cpu")
    lenet_data = torch.load("lenet_stego_480bits"+str(i)+".pth")
    lenet.load_state_dict(lenet_data.state_dict())
    gcn_out = gcn_model(tensor_adjacency, lenet[0], lenet[3], 0)
    gcn_out = gcn_out[6:, :]
    Y_decision = decision(gcn_out)
    Y_decision = Y_decision.reshape([1, -1])
    decoded = Y_decision.detach().numpy().tolist()[0]
    result_binarys = list(map(int, decoded))
    result_str = ascii2str([result_binarys])
    print(result_str,end="\n")

    """
        KL Divergence
    """

    divergence = calculate_KL_divergence(origin_net, lenet)
    print("KL Divergence of ", "\"lenet_stego_480bits"+str(i)+".pth\" is:" , divergence)

    """
        add some noise
    """
    with open("lenet_junyun_480bit"+str(i)+".txt", "w") as f:
        for j in range(1, 31):
            j = j * 0.01
            gcn_out = gcn_model(tensor_adjacency, lenet[0], lenet[3], j)
            gcn_out = gcn_out[6:, :]
            Y_decision = decision(gcn_out)
            correct_rate, error_rate = gcn_accuracy(Y_decision, secret_tensor)
            f.write(str(error_rate) + '\n')





