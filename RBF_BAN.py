import torch
import torch.nn as nn
import torch.nn.init as init

class RBFBANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=0.5, beta=1.0):
        super(RBFBANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.beta = beta
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        init.xavier_uniform_(self.weights)

    def yukawa_rbf(self, distances):
        return (self.beta / distances) * torch.exp(-self.alpha * distances)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.yukawa_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output