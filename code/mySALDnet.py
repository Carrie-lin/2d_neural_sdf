import numpy as np
import torch
import torch.nn as nn

# ========================================== network ===================================================================
class Decoder(nn.Module):
    def __init__(
        self,
        weight_norm=False,
    ):
        super().__init__()
        dims=[100,100,100,98,100,100,100,100]
        dims = [2] + dims + [1]
        self.num_layers = len(dims)
        self.norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        self.xyz_in_all = False
        self.weight_norm = True
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            if l==4:
                lin = nn.Linear(dims[l+1],dims[l+1])
            else:
                lin = nn.Linear(dims[l], out_dim)
            # pay attention
            '''
            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.000001)
                torch.nn.init.constant_(lin.bias, -0.4)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm and l in self.norm_layers:
                lin = nn.utils.weight_norm(lin) # pay attention
            '''
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.Softplus(beta=100)

    def forward(self, input):
        x = input
        skip_input=input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l==4:
                x= torch.cat([skip_input, x], 1)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x

class SALNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()

    def forward(self, non_mnfld_pnts):
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1]))
        return nonmanifold_pnts_pred

# ========================================== network ===================================================================
