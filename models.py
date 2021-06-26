import megengine.module as M
import megengine.functional as F

class Net(M.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = M.Conv2d(16, 256, kernel_size=(2, 1),
                              stride=1, padding=0)
        self.relu0 = M.ReLU()
        self.conv1 = M.Conv2d(256, 256, kernel_size=(1, 2),
                              stride=1, padding=0)
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(256, 256, kernel_size=(2, 1),
                              stride=1, padding=0)
        self.relu2 = M.ReLU()
        self.conv3 = M.Conv2d(256, 256, kernel_size=(1, 2),
                              stride=1, padding=0)
        self.relu3 = M.ReLU()
        self.fc1 = M.Linear(1024, 16)
        self.relu5 = M.ReLU()
        # self.fc2 = M.Linear(16, 3)
        self.fc2 = M.Linear(16, 4)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = F.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = F.reshape(x, (-1, 4))
        return x


class WideNet(M.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        # layer1
        self.conv1 = M.Conv2d(16, 128, kernel_size=(2, 1), stride=1, padding=0)
        self.conv2 = M.Conv2d(16, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.relu1 = M.ReLU()
        self.relu2 = M.ReLU()

        # layer2
        self.conv11 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.conv12 = M.Conv2d(128, 128, kernel_size=(2, 1), stride=1, padding=0)
        self.conv21 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.conv22 = M.Conv2d(128, 128, kernel_size=(2, 1), stride=1, padding=0)

        self.relu11 = M.ReLU()
        self.relu12 = M.ReLU()
        self.relu21 = M.ReLU()
        self.relu22 = M.ReLU()

        #
        self.fc1 = M.Linear(7424, 256)
        self.relu = M.ReLU()
        self.A = M.Linear(256, 4)
        self.V = M.Linear(256, 1)


    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x))

        x11 = self.relu11(self.conv11(x1))
        x12 = self.relu12(self.conv12(x1))
        x21 = self.relu21(self.conv21(x2))
        x22 = self.relu22(self.conv22(x2))

        x1 = F.flatten(x1, 1)
        x2 = F.flatten(x2, 1)
        x11 = F.flatten(x11, 1)
        x12 = F.flatten(x12, 1)
        x21 = F.flatten(x21, 1)
        x22 = F.flatten(x22, 1)

        hidden = F.concat([x1, x2, x11, x12, x21, x22], axis=1)
        hidden = self.relu(self.fc1(hidden))
        action = self.A(hidden)
        value = self.V(hidden)
        
        Q = value + (action - action.mean(axis=1, keepdims=True))
        return Q