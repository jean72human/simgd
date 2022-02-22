import torch
import torch.nn as nn

class GCNModel(nn.Module):
    def __init__(self, num_layers=2, sz_in=1433, sz_hid=32, sz_out=7):
        super().__init__()

        ### BEGIN SOLUTION
        self.hidden1 = nn.Linear(sz_in,sz_hid, bias=False)
        self.out1 = nn.Linear(sz_hid,sz_out, bias=False)
        self.hidden2 = nn.Linear(sz_in,sz_hid, bias=False)
        self.out2 = nn.Linear(sz_hid,sz_out, bias=False)
        self.ReLU = nn.ReLU()
        ### END SOLUTION

    def forward(self,  x, edges, node_graph_ind):
        """
        Args:
            fts: [N, 1433]
            adj: [N, N]

        Returns:
            new_fts: [N, 7]
        """
        adj1 = torch.zeros((x.size(0),x.size(0)), device=x.device)
        for ind in edges[:,:2]: 
            adj1[ind[0],ind[1]]=1
            #adj[ind[1],ind[0]]=1

        deg1 = adj1.sum(axis=1, keepdim=True) # Degree of nodes, shape [N, 1]
        adj2 = adj1.T
        deg2 = adj2.sum(axis=1, keepdim=True)

        ### BEGIN SOLUTION
        x1 = self.ReLU(self.hidden1(1/deg1 * adj1 @ x))
        x1 = self.out1(1/deg1 * adj1 @ x1)

        x2 = self.ReLU(self.hidden2(1/deg2 * adj2 @ x))
        x2 = self.out2(1/deg1 * adj2 @ x2)
        ### END SOLUTION
        return (x1 + x2)/2
