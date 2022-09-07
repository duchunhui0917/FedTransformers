import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

v = 0.9


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        w = torch.empty((20, 768))
        nn.init.xavier_uniform_(w)
        self.x = nn.Parameter(w)

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def forward(self):
        sim = self.sim(self.x, self.x)
        num = sim.shape[0]
        loss = 0.
        for i in range(num):
            for j in range(num):
                if j == i:
                    continue
                score = max([0, v - sim[i][j]]) ** 2
                loss += score
        return sim, loss


model = MyModule()
print(list(model.named_parameters()))
optimizer = optim.AdamW(model.parameters(), lr=0.1)
for i in range(100):
    sim, loss = model()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(sim, loss)
print(list(model.named_parameters()))
