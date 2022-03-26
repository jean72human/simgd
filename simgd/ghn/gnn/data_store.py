import torch

class DataStore:
    def __init__(self, data):
        self.data = data


    def __getitem__(self,idx):
        return self.data[idx]
    
    def __setitem__(self,key,value):
        self.data[key] = value

    def __radd__(self,values):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key] + values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values
        return to_return

    def __iadd__(self,values):
        for key in self.data:
            self.data[key] = self.data[key] + values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values


    def __rmul__(self,values):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key] * values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values
        return to_return

    def __imul__(self,values):
        for key in self.data:
            self.data[key] = self.data[key] * values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values

    
    def __rsub__(self,values):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key] - values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values
        return to_return

    def __isub__(self,values):
        for key in self.data:
            self.data[key] = self.data[key] - values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values


    def __rtruediv__(self,values):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key] / values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values
        return to_return

    def __itruediv__(self,values):
        for key in self.data:
            self.data[key] = self.data[key] / values[key] if isinstance(values,dict) or isinstance(values,DataStore) else values
    


    def detach(self):
        for key in self.data:
            self.data[key].detach()

    def norm(self):
        to_return = {}
        for key in self.data:
            to_return[key] = self.data[key].norm()
        return to_return 

    def normalize(self):
        for key in self.data:
            self.data[key] /= self.data[key].norm()

    def to(self,device):
        for key in self.data:
            self.data[key].to(device)


