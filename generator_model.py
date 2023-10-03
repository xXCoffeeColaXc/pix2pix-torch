import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()

        # refactor it to Encoder and Decoder 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect") 
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.down = down


    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        # first layer, no BatchNorm
        # ReLU = LeakyReLU with a slope 0.2
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #128
            nn.LeakyReLU(0.2)
        )

        # encoder: C64-C128-C256-C512-C512-C512-C512-C512
        self.down1 = Block(features  , features*2, down=True, act="leaky", use_dropout=False)#64
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False)#32
        self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=False)#16
        self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)#8
        self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)#4
        self.down6 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)#2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #1
            nn.ReLU()    
        )
        
        # decoder: CD512-CD512-CD512-C512-C256-C128-C64
        # in_channels*2: x,y <- concatenate these along the channels
        self.up1 = Block(features*8  , features*8, down=False, act="relu", use_dropout=True) #2
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) #4
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) #8
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False)#16
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False)#32
        self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False)#64
        self.up7 = Block(features*2*2, features  , down=False, act="relu", use_dropout=False)#128
        
        # after last layer conv layer is applied to map to the number of output channels (3), followed by Tanh function
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1), #256
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        # CD512-CD1024-CD1024-CD1024-CD512-C256-C128 
        # skip connection:concatenate activations from layer i to layer n-i
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final_up(torch.cat([up7, d1], dim=1))
    
def test():
    x = torch.randn((1,3,256,256))
    model = Generator()
    preds = model(x)
    print(preds.shape)
    print(model.parameters)

if __name__ == "__main__":
    test()