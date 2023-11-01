import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, outputs, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        #layer to estimate the state value function 
        self.fc1_value = nn.Linear(7*7*64, 512)
        self.fc1_out_value = nn.Linear(512, 1)
        #layer to estimate the advantage function
        self.fc2_advantage = nn.Linear(7*7*64, 512)
        self.fc2_out_advantage = nn.Linear(512, outputs)
        
        self.device = device
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        
        """Forward method implementation."""
        #value and advantage streams
        value = F.relu(self.fc1_value(x)) 
        advantage = F.relu(self.fc2_advantage(x))

        value = self.fc1_out_value(value)
        advantage = self.fc2_out_advantage(advantage)
        
        #combining value function and advantage function to get the q values
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q
