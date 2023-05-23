import torch 
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(device)
    print(torch.cuda.device_count())