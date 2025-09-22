class AA:
    def __init__(self):
        print("init")
    
    def run(self):
        print("run")
        
        
import torch

a=torch.rand(2,3,2)
check=torch.rand(2,3,1)

res=a<check
print(f"a:{a}")
print(f"check:{check}")
print(res)
