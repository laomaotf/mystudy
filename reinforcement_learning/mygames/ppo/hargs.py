import yaml

class HARGS:
    def __init__(self, path) -> None:
        with open(path,'r') as f:
            self.data = yaml.load(f,Loader=yaml.FullLoader)
        return
    def get(self,key,default_value=None):
        if key in self.data.keys():
            return self.data[key]
        print("not set hyper-param : {}".format(key))
        return default_value