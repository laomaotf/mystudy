import yaml
import traceback




class CONFIG:
    def __init__(self,config_file = None):
        self.data = None
        if config_file is not None:
            self.load(config_file)
        pass
    def load(self,config_file):
        try:
            with open(config_file,'r') as f:
                self.data = yaml.safe_load(f)
        except:
            traceback.print_exc()
        return True
    def __call__(self, keys):
        return self.get(keys)    
    def get(self,keys):
        try:
            keys = keys.split('.')
            value = self.data[keys[0]]
            for key in keys[1:]:
                value = value[key]
        except:
            traceback.print_exc()
        return value
            
if __name__ == "__main__":    
    import os        
    cfg = CONFIG(os.path.join(os.path.dirname(__file__),'pvz_easy_00.yaml'))
    print(cfg("action"))
                
        
        