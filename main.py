import torch 
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count())