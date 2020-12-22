import torch
import torch.nn as nn
import torch.nn.functional as F

###LeNet
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, (5, 5),
                                     padding=2)  # output shape [batch_size x 6 x input_width x input_height]
        self.conv2 = torch.nn.Conv2d(6, 16, (5, 5))  # output [b_size x 16 x 60 x 60] if input is bsize x 3 x 64 x 64
        self.fc1 = torch.nn.Linear(16 * 14 * 14, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print(x.shape)
        x = x.view(x.shape[0], self.num_flat_features(x))
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LeNet5(torch.nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        print(f"after convs output has shape {x.shape}")
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs



"""### Modified Densenet

Conv2d arithmetic:

`OutputImgDim = [(InputImgDim−K+2P)/S]+1`

K = Kernel size, P = padding, S = stride
"""


class Block(nn.Module):
    def __init__(
            self, in_channels, interm_channels: list, identity_downsample=None, stride=1, device = torch.device('cpu')
    ):
        super(Block, self).__init__()
        self.device = device
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels, interm_channels[0], kernel_size=3, stride=1, padding=1  # padding 1 retains spatial dimensions
        )])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(interm_channels[0])])
        # self.expansion = 4
        for i in range(len(interm_channels) - 1):  # each block has 5 conv layers by choice
            self.convs.append(nn.Conv2d(
                interm_channels[i], interm_channels[i + 1], kernel_size=3, stride=1, padding=1
            ))
            self.batch_norms.append(nn.BatchNorm2d(interm_channels[i + 1]))
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        # print(f"when entering through block: {torch.cuda.device_of(x)}")
        identity = x.clone().to(self.device)
        # print(f"after cloned in block: {torch.cuda.device_of(x)}")
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            # print(f"in block: shape after conv {i}: {x.shape}")
            x = self.batch_norms[i](x)
            # print(f"in block: shape after batch norm {i}: {x.shape}")
            x = self.relu(x)

        x = self.maxpool(x)
        # print(f"in block: shape after maxpool: {x.shape}")
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # print(f"in block: shape of downsampled identity: {identity.shape}")
        x = torch.cat((x, identity), dim=1)  # concatenate along channel dimension, which is 1
        # print(f"in block: shape after concat: {x.shape}")
        x = self.relu(x)
        return x


# #TEST TEST TEST
# batch = torch.randn(11,3,64,64).to(device)
# identity_downsample = nn.Conv2d(
#                 in_channels = 3,
#                 out_channels = 3, #input and output channels stay fixed, because we concatenate on channel dimension
#                 kernel_size = 11,
#                 stride=1
#                 )
# block = Block(3, [2**i for i in range(5,10)], identity_downsample)
# block = block.to(device)
# print(torch.cuda.device_of(block(batch)))

class ModDenseNet(nn.Module):
    def __init__(self, block, image_channels, num_classes, device):
        super(ModDenseNet, self).__init__()
        self.device = device
        self.relu = nn.ReLU()
        self.image_channels = image_channels
        self.block1 = self._make_block(
            block, in_channels=self.image_channels, interm_channels=[2 ** i for i in range(5, 10)]
        )
        channels_added_concat = self.image_channels * 2 ** 2
        self.block2 = self._make_block(
            block, in_channels=2 ** 9 + channels_added_concat, interm_channels=[2 ** i for i in range(6, 11)]
        )
        channels_added_concat = (2 ** 9 + channels_added_concat) * 2 ** 2
        self.block3 = self._make_block(
            block, in_channels=2 ** 10 + channels_added_concat,
            interm_channels=[2 ** 5] + [2 ** i for i in range(7, 11)]
        )
        channels_added_concat = (2 ** 10 + channels_added_concat) * 2 ** 2
        self.last_conv = nn.Conv2d(in_channels=2 ** 10 + channels_added_concat, out_channels=200, kernel_size=1,
                                   stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # preserves channels, output has HxW=1x1
        self.fc = nn.Linear(200, num_classes)

    def forward(self, x):
        # print(f"after entering through model: {torch.cuda.device_of(x)}")
        x = self.block1(x)
        # print(f"after gone through block 1: {torch.cuda.device_of(x)}")
        # print(f"shape after block1: {x.shape}")
        x = self.block2(x)
        # print(f"shape after block2: {x.shape}")
        x = self.block3(x)
        # print(f"shape after block3: {x.shape}")
        x = self.last_conv(x)
        # print(f"shape after last conv layer: {x.shape}")
        x = self.relu(x)

        x = self.avgpool(x)
        # print(f"shape after avgpool: {x.shape}")
        x = x.reshape(x.shape[0], -1)
        # print(f"shape after reshape before fully connected: {x.shape}")
        x = self.fc(x)
        # print(f"shape after reshape after fully connected: {x.shape}")

        return x

    def _make_block(self, block, in_channels, interm_channels):
        # need to downsample to adjust (just) for a modification of 1 maxpool with stride ker 2x2 stride 2 padding 0
        block_size = 2

        def space_to_depth(x):
            """
            Unfolded maintains batch dimension. It takes (c, h, w) and gives output of (a, b), where:
            - a = k x k x c i.e. values contained in each "patch" (kernel) but also counting channel dims
            - b = ([(h-k)/s]+1) x ([(w-k)/s]+1) i.e. how many kernels of given shape, also taking into account stride (and also padding etc. but assumed padding = 0) "fit into" spacial region
            """
            n, c, h, w = x.size()
            unfolded_x = torch.nn.functional.unfold(x, kernel_size=block_size, stride=block_size)
            unfolded_x = unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)
            return unfolded_x

        identity_downsample = space_to_depth
        return block(in_channels, interm_channels, identity_downsample, stride = 1, device = self.device)

# batch = torch.randn(20,3,64,64).to(device)
# model = ModDenseNet(block = Block, image_channels = 3, num_classes = 10).to(device)
# print(model(batch).shape)