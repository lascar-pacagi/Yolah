import sys
from features_network_mono import Net, FeaturesDataset, LAST_MODEL
import torch

device = "cpu"
net = Net()
net.load_state_dict(torch.load(LAST_MODEL, map_location=torch.device('cpu')))
net.to(device)
net.eval()

def get_prediction(features):
    features = features.unsqueeze(0).to(device)
    logits = net(features)[0]
    black, draw, white = logits[0].item(), logits[1].item(), logits[2].item()
    return black, draw, white

def main():
    dataset = FeaturesDataset(sys.argv[1])
    for i in range(len(dataset)):                        
        black_proba, draw_proba, white_proba = get_prediction(dataset[i][0])
        print(f'{black_proba:.17}\n{draw_proba:.17}\n{white_proba:.17}\n')      

if __name__ == "__main__":
    main()

