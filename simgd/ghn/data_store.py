import torch

class DataStore:
    def __init__(self, data):
        self.data = data


    def __getitem__(self,idx):
        return self.data[idx]
    
    def __setitem__(self,key,value):
        self.data[key] = value

    def __add__(self,values):
        to_return = {}
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            to_return[key] = self.data[key] + op
        return DataStore(to_return) 

    def __iadd__(self,values):
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            self.data[key] = self.data[key] + op
        return self


    def __mul__(self,values):
        to_return = {}
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            to_return[key] = self.data[key] * op
        return DataStore(to_return) 

    def __imul__(self,values):
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            self.data[key] = self.data[key] * op
        return self

    
    def __sub__(self,values):
        to_return = {}
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            to_return[key] = self.data[key] - op
        return DataStore(to_return) 

    def __isub__(self,values):
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            self.data[key] = self.data[key] - op
        return self


    def __truediv__(self,values):
        to_return = {}
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            to_return[key] = self.data[key] / op
        return DataStore(to_return) 

    def __itruediv__(self,values):
        for key in self.data:
            op = values[key] if (isinstance(values,dict) or isinstance(values,DataStore)) else values
            self.data[key] = self.data[key] / op
        return self

    def values(self):
        return self.data.values()

    def keys(self):
        return self.data.keys()
    


    def detach(self):
        for key in self.data:
            self.data[key] = self.data[key].detach()
        return self

    def norm(self):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key].norm()
        return DataStore(to_return) 

    def normalize(self):
        for key in self.data:
            self.data[key] /= self.data[key].norm()
        return self

    def to(self,device):
        for key in self.data:
            self.data[key].to(device)
        return self

    def clone(self):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key].clone()
        return DataStore(to_return)

    @staticmethod
    def empty_dict(net, graph):
        out = {}
        for ind, (name,param) in enumerate(graph.node_params[1:]):
            if param in net.state_dict().keys():
                out[param] = None
        return DataStore(out)


