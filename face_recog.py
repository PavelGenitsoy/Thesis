import torch
import torchvision.models as models
import torch.nn as nn

from torchsummary import summary


models_dict = {
    'alexnet': models.alexnet,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg11_bn': models.vgg11_bn,
    'vgg13_bn': models.vgg13_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet1_0': models.mnasnet1_0,
    'shufflenetv2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenetv2_x1_0': models.shufflenet_v2_x1_0,
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_8gf': models.regnet_y_8gf,
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_y_32gf': models.regnet_y_32gf,
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_800mf': models.regnet_x_800mf,
    'regnet_x_1_6gf': models.regnet_x_1_6gf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_x_16gf': models.regnet_x_16gf,
    'regnet_x_32gf': models.regnet_x_32gf,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5
}


def reset_weights(m):
    if m._get_name() == 'SqueezeNet':
        layer_to_reset = list(m.children())[-1][1]
    else:
        layer_to_reset = list(m.children())[-1][-1]
    if hasattr(layer_to_reset, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer_to_reset}')
        layer_to_reset.reset_parameters()


class FaceRecog(nn.Module):
    def __init__(self, num_classes, model_name: str, pretrained=True):
        super().__init__()

        self.model = models_dict[model_name](pretrained=pretrained)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        if isinstance(list(self.model.children())[-1], nn.Linear):
            if self.model._get_name() == 'ShuffleNetV2':
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(list(self.model.children())[-1].in_features, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(list(self.model.children())[-1].in_features, num_classes)
                )
        elif isinstance(list(self.model.children())[-1], nn.Sequential):
            if isinstance(list(self.model.children())[-1][-1], nn.Linear):
                if self.model._get_name() in ('MobileNetV2', 'MobileNetV3', 'MNASNet', 'EfficientNet'):
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        nn.Flatten(),
                        *list(self.model.children())[-1][:-1],
                        nn.Linear(list(self.model.children())[-1][-1].in_features, num_classes)
                    )
                elif self.model._get_name() == 'ConvNeXt':
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        *list(self.model.children())[-1][:-1],
                        nn.Linear(list(self.model.children())[-1][-1].in_features, num_classes)
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(6, 6) if self.model._get_name() == 'AlexNet' else (7, 7)),
                        nn.Flatten(),
                        *list(self.model.children())[-1][:-1],
                        nn.Linear(list(self.model.children())[-1][-1].in_features, num_classes)
                    )
            elif isinstance(list(self.model.children())[-1][-1], nn.AdaptiveAvgPool2d):
                if self.model._get_name() == 'SqueezeNet':
                    self.classifier = nn.Sequential(
                        nn.Dropout(p=0.5, inplace=False),
                        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
                        *list(self.model.children())[-1][-2:],
                        nn.Flatten()
                    )
                else:
                    raise "###### ERROR: unknown NN where last layer from classifier is AdaptiveAvgPool2d ######"
            else:
                raise "###### ERROR: unknown type of last layer from classifier ######"
        else:
            raise "###### ERROR: unknown type of last element from model.children() ######"

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        y = self.classifier(x)
        return y

    def summary(self, input_size):
        return summary(self, input_size)
