import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512]) -> None:
        super().__init__()

        # dont we need weight init? weight_init: Gaussian(0,0.02), 

        # first layer, no BatchNorm
        # ReLU = LeakyReLU with a slope 0.2
        self.initial = nn.Sequential(
            # in_channels*2: x,y <- concatenate these along the channels
            nn.Conv2d(in_channels*2 , features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # C64-C128-C256-C512
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
            # sigmoid ?
        )
        
        self.model = nn.Sequential(*layers)


    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
'''
input: 3,256,256
layers: 
6,64    stride=2    padding=1   img_size=128
64,128  stride=2    padding=1   img_size=64
128,256 stride=2    padding=1   img_size=32
256,512 stride=1    padding=1   img_size=31
512,1   stride=1    padding=1   img_size=30
'''

def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape)
    print(model)

if __name__ == "__main__":
    test()

