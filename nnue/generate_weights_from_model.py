from nnue_multigpu import Net, LAST_MODEL
import torch
from collections import OrderedDict

def write_parameters(file, parameters):
    values = [str(p.item()) for p in parameters]
    file.write("\n".join(values) + "\n")

def save_model_parameters(filename, model):    
    with open(filename, "wt") as file:
        for p in model.parameters():
            assert p.dim() in [1, 2]            
            if p.dim() == 1:
                file.write(f"B\n{len(p.data)}\n")
                write_parameters(file, p.data)
            else:
                n_rows = len(p.data)
                n_cols = len(p.data[0])                
                file.write(f"W\n{n_rows}\n{n_cols}\n")                
                for row in p.data:
                    write_parameters(file, row)

if __name__ == "__main__":
    net = Net()
    net.to('cpu')
    state_dict = torch.load(LAST_MODEL, map_location=torch.device('cpu'))
    net_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        net_dict[name] = v
    net.load_state_dict(net_dict)
    #net.load_state_dict(state_dict)
    save_model_parameters("nnue_parameters.txt", net)
    torch.save(net_dict, "nnue.pt")
