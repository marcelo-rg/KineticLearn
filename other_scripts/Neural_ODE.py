import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['rk4', 'adams'], default='rk4')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[1., 1., 1.05]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


true_A2 = torch.tensor([[-0.1, -0.5], [0.5, -0.1]])
true_B2 = torch.tensor([[0.2, 1.], [-1, 0.2]])

class Lambda2(nn.Module):
    
    def __init__(self, A, B):
        super(Lambda2, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
    
    def forward(self, t, y):
        xTx0 = torch.sum(y * true_y0, dim=1)
        dxdt = torch.sigmoid(xTx0) * self.A(y - true_y0) + torch.sigmoid(-xTx0) * self.B(y + true_y0)
        return dxdt
    

class Lorenz(nn.Module):

    def lorenz(self, xyz, *, s=13, r=33, b=8/3):

        x, y, z = xyz[0]
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])
    
    def forward(self, t, y):

        return torch.tensor(self.lorenz(y))

class Chem(nn.Module):

    def chem(self, xyz, *, k1=0.1, k2=0.5, k3=0.2):

        x, y, z = xyz[0]
        x_dot = (k1-k2)*x + k2*y
        y_dot = k1*x - k2*y
        z_dot = k3*x
        return np.array([x_dot, y_dot, z_dot])
    
    def forward(self, t, y):

        return torch.tensor(self.chem(y))

with torch.no_grad():
    true_y = odeint(Lorenz(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

diraddress = "C:\\Users\\clock\\Desktop\\Python\\Images\\NeuralODEs\\png_non_linear"
if args.viz:
    makedirs(diraddress)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, projection='3d')
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y,z')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1],'--',t.cpu().numpy(),pred_y.cpu().numpy()[:, 0, 2], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(0, 10)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        # ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        # ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0],true_y.cpu().numpy()[:, 0, 1],true_y.cpu().numpy()[:, 0, 2], lw=0.8)
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0],pred_y.cpu().numpy()[:, 0, 1],pred_y.cpu().numpy()[:, 0, 2],"--", lw=0.8)
        ax_phase.set_xlabel("X Axis")
        ax_phase.set_ylabel("Y Axis")
        ax_phase.set_zlabel("Z Axis")
        # ax_phase.set_xlim(0, 10)
        # ax_phase.set_ylim(0, 10)
        # ax_phase.set_zlim(0, 10)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)


        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig(diraddress+'/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)




class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 3),
            # nn.Tanh(),
            # nn.Linear(20, 20),
            # nn.Tanh(),
            # nn.Linear(20, 3),

        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y)**2)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()