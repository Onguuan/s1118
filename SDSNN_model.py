import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5
tau = 0.25
a = 1

cfg_cnn_fashion = [(1, 64, 1, 1, 3),
                   (64, 256, 1, 1, 3),
                   (256, 512, 1, 1, 3),
                   (512, 10, 1, 1, 3)]

cfg_cnn_cifar10 = [(3, 128, 1, 1, 3),
                   (128, 256, 1, 1, 3),
                   (256, 512, 1, 1, 3),
                   (512, 1024, 1, 1, 3),
                   (1024, 512, 1, 1, 3),
                   (512, 10, 1, 1, 3)]

cfg_cnn_cifar100 = [(3, 128, 1, 1, 3),
                   (128, 256, 1, 1, 3),
                   (256, 512, 1, 1, 3),
                   (512, 1024, 1, 1, 3),
                   (1024, 512, 1, 1, 3),
                   (512, 100, 1, 1, 3)]


cfg_imgsize_fashion = [28, 14, 7, 6, 5]
cfg_imgsize_cifar = [32, 16, 8, 4, 2, 1]

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pre_input, t):
        if torch.all(pre_input == 0):
            x = input.gt(thresh).float()
        else:
            x = torch.where((pre_input > input) * (input > thresh / t),
                            torch.min(3 * torch.ones_like(input), torch.floor(input.div(thresh / t))),
                            torch.zeros_like(input).to(device))
        if x.sum().item() == 0:
            x = torch.where((input > thresh),
                            torch.min(3 * torch.ones_like(input), torch.floor(input.div(thresh / t))),
                            torch.zeros_like(input).to(device))
        ctx.save_for_backward(input, torch.tensor(t))
        return x
    @staticmethod
    def backward(ctx, grad_output):
        input, t, = ctx.saved_tensors
        if t == 0:
            t = 1
        grad_input = grad_output.clone()
        temp = abs(input - thresh / t) < 1 / 2
        return temp * grad_input.float(), None, None

act_fun = ActFun.apply

def mem_update(x, mem_u, output, t):
    mem_u_new = torch.where(output != 0, mem_u * tau - thresh, mem_u * tau)
    mem_u_new = mem_u_new + x
    output = act_fun(mem_u_new, mem_u, t)
    return mem_u_new, output


class SCNN_Fashion_MNIST(nn.Module):
    def __init__(self, batch_size, t1, t2, t3, t4):
        super(SCNN_Fashion_MNIST, self).__init__()
        self.batch_size = batch_size
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_fashion[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_fashion[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.avg_pool2d1 = nn.AvgPool2d(2)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_fashion[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_fashion[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(10)

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, cfg_cnn_fashion[0][1], cfg_imgsize_fashion[0],
                                        cfg_imgsize_fashion[0], device=device)
        c2_mem = c2_spike = torch.zeros(self.batch_size, cfg_cnn_fashion[1][1], cfg_imgsize_fashion[1],
                                        cfg_imgsize_fashion[1], device=device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, cfg_cnn_fashion[2][1], cfg_imgsize_fashion[2],
                                        cfg_imgsize_fashion[2], device=device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, cfg_cnn_fashion[3][1], cfg_imgsize_fashion[2],
                                        cfg_imgsize_fashion[2], device=device)

        c1_1 = self.conv1(input.float())
        c1_2 = self.bn1(c1_1)
        for t in range(int(self.t1)):
            c1_mem, c1_spike = mem_update(c1_2, c1_mem, c1_spike, t)

        c2_1_ = self.MaxPool2d(c1_spike)
        c2_1 = self.conv2(c2_1_)
        c2_2 = self.bn2(c2_1)
        for t in range(int(self.t2)):
            c2_mem, c2_spike = mem_update(c2_2, c2_mem, c2_spike, t)

        c3_1_ = self.MaxPool2d(c2_spike)
        c3_1 = self.conv3(c3_1_)
        c3_2 = self.bn3(c3_1)
        for t in range(int(self.t3)):
            c3_mem, c3_spike = mem_update(c3_2, c3_mem, c3_spike, t)

        c4_1 = self.conv4(c3_spike)
        c4_2 = self.bn4(c4_1)
        for t in range(int(self.t4)):
            c4_mem, c4_spike = mem_update(c4_2, c4_mem, c4_spike, t)

        x = self.global_avg_pool2d(c4_spike)
        x_sq = x.squeeze(2).squeeze(2)

        return x_sq


class SCNN_cifar10(nn.Module):
    def __init__(self, batch_size, t1, t2, t3, t4, t5, t6):
        super(SCNN_cifar10, self).__init__()
        self.batch_size = batch_size
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar10[5]
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.MaxPool2d = nn.MaxPool2d(2, 2)

        self.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(100)

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[0][1], cfg_imgsize_cifar[0],
                                        cfg_imgsize_cifar[0],
                                        device=device)
        c2_mem = c2_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[1][1], cfg_imgsize_cifar[0],
                                        cfg_imgsize_cifar[0],
                                        device=device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[2][1], cfg_imgsize_cifar[1],
                                        cfg_imgsize_cifar[1],
                                        device=device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[3][1], cfg_imgsize_cifar[2],
                                        cfg_imgsize_cifar[2],
                                        device=device)
        c5_mem = c5_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[4][1], cfg_imgsize_cifar[2],
                                        cfg_imgsize_cifar[2],
                                        device=device)
        c6_mem = c6_spike = torch.zeros(self.batch_size, cfg_cnn_cifar10[5][1], cfg_imgsize_cifar[2],
                                        cfg_imgsize_cifar[2],
                                        device=device)

        c1_1 = self.conv1(input.float())
        c1_2 = self.bn1(c1_1)
        for t in range(int(self.t1)):
            c1_mem, c1_spike = mem_update(c1_2, c1_mem, c1_spike, t)

        c2_1 = self.conv2(c1_spike)
        c2_2 = self.bn2(c2_1)
        for t in range(int(self.t2)):
            c2_mem, c2_spike = mem_update(c2_2, c2_mem, c2_spike, t)

        c3_1_ = self.MaxPool2d(c2_spike)
        c3_1 = self.conv3(c3_1_)
        c3_2 = self.bn3(c3_1)

        for t in range(int(self.t3)):
            c3_mem, c3_spike = mem_update(c3_2, c3_mem, c3_spike, t)

        c4_1_ = self.MaxPool2d(c3_spike)
        c4_1 = self.conv4(c4_1_)
        c4_2 = self.bn4(c4_1)
        for t in range(int(self.t4)):
            c4_mem, c4_spike = mem_update(c4_2, c4_mem, c4_spike, t)

        c5_1 = self.conv5(c4_spike)
        c5_2 = self.bn5(c5_1)
        for t in range(int(self.t5)):
            c5_mem, c5_spike = mem_update(c5_2, c5_mem, c5_spike, t)

        c6_1 = self.conv6(c5_spike)
        c6_2 = self.bn6(c6_1)
        for t in range(int(self.t6)):
            c6_mem, c6_spike = mem_update(c6_2, c6_mem, c6_spike, t)

        x_b = self.global_avg_pool2d(c6_spike)

        return x_b.squeeze()



class SCNN_cifar100(nn.Module):
    def __init__(self, batch_size, t1, t2, t3, t4, t5, t6):
        super(SCNN_cifar100, self).__init__()
        self.batch_size = batch_size
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn_cifar100[5]
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, )

        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(100)

    def forward(self, input, batch_size=8):
        c1_mem = c1_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[0][1], cfg_imgsize_cifar[0],
                                        cfg_imgsize_cifar[0],
                                        device=device)
        c2_mem = c2_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[1][1], cfg_imgsize_cifar[0],
                                        cfg_imgsize_cifar[0],
                                        device=device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[2][1], cfg_imgsize_cifar[1],
                                        cfg_imgsize_cifar[1],
                                        device=device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[3][1], cfg_imgsize_cifar[2],
                                        cfg_imgsize_cifar[2],
                                        device=device)
        c5_mem = c5_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[4][1], cfg_imgsize_cifar[2],
                                        cfg_imgsize_cifar[2],
                                        device=device)
        c6_mem = c6_spike = torch.zeros(self.batch_size, cfg_cnn_cifar100[5][1], cfg_imgsize_cifar[3],
                                        cfg_imgsize_cifar[3],
                                        device=device)

        conv1_out = self.conv1(input.float())
        bn1_out = self.bn1(conv1_out)
        for t in range(self.t1):
            c1_mem, c1_spike = mem_update(bn1_out, c1_mem, c1_spike, t)

        conv2_out = self.conv2(c1_spike)
        bn2_out = self.bn2(conv2_out)
        for t in range(self.t2):
            c2_mem, c2_spike = mem_update(bn2_out, c2_mem, c2_spike, t)

        x1 = self.MaxPool2d(c2_spike)
        conv3_out = self.conv3(x1)
        bn3_out = self.bn3(conv3_out)
        for t in range(self.t3):
            c3_mem, c3_spike = mem_update(bn3_out, c3_mem, c3_spike, t)

        x2 = self.MaxPool2d(c3_spike)
        conv4_out = self.conv4(x2)
        bn4_out = self.bn4(conv4_out)
        for t in range(self.t4):
            c4_mem, c4_spike = mem_update(bn4_out, c4_mem, c4_spike, t)

        conv5_out = self.conv5(c4_spike)
        bn5_out = self.bn5(conv5_out)
        for t in range(self.t5):
            c5_mem, c5_spike = mem_update(bn5_out, c5_mem, c5_spike, t)

        x3 = self.MaxPool2d(c5_spike)
        conv6_out = self.conv6(x3)
        bn6_out = self.bn6(conv6_out)
        for t in range(self.t6):
            c6_mem, c6_spike = mem_update(bn6_out, c6_mem, c6_spike, t)

        x_b = self.global_avg_pool2d(c6_spike)

        return x_b.squeeze()



def Fashion_MNIST(batch_size, t1, t2, t3, t4):
    model = SCNN_Fashion_MNIST(batch_size, t1, t2, t3, t4)
    return model

def cifar10(batch_size, t1, t2, t3, t4, t5, t6):
    model = SCNN_cifar10(batch_size, t1, t2, t3, t4, t5, t6)
    return model

def cifar100(batch_size, t1, t2, t3, t4, t5, t6):
    model = SCNN_cifar100(batch_size, t1, t2, t3, t4, t5, t6)
    return model
