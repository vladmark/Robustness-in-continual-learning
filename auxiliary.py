def get_model(model_type, cl_dset, device):
  if model_type == "LeNet":
    from Models import LeNet
    model = LeNet()
  elif model_type == "LeNet5":
    from Models import LeNet5
    model = LeNet5()
  elif model_type == "ModDenseNet":
    from Models import ModDenseNet
    from Models import Block
    model = ModDenseNet(block = Block, image_channels = 3, num_classes = cl_dset['meta']['cls_no'], device = device)
  elif "Resnet" in model_type:
    import torchvision
    import torch
    if model_type == "Resnet152":
      model = torchvision.models.resnet152(pretrained=False)
      model.__class__.__name__ = "ResNet152"
    elif model_type == "Resnet101":
      model = torchvision.models.resnet101(pretrained=False)
      model.__class__.__name__ = "ResNet101"
    elif model_type == "Resnet50":
      model = torchvision.models.resnet50(pretrained=False)
      model.__class__.__name__ = "ResNet50"
    elif model_type == "Resnet18":
      model = torchvision.models.resnet18(pretrained=False)
      model.__class__.__name__ = "ResNet18"
    elif model_type == "Densenet169":
      model = torchvision.models.densenet169(pretrained=False)
      model.__class__.__name__ = "DenseNet169"
    elif model_type == "Densenet201":
      model = torchvision.models.densenet201(pretrained=False)
      model.__class__.__name__ = "ResNet201"
    model.fc = torch.nn.Sequential(
              torch.nn.Linear(in_features = model.fc.in_features, out_features = model.fc.in_features//4, bias = True), #was initially /2 -> /4 -> /8
              torch.nn.ReLU(),
              torch.nn.Linear(in_features = model.fc.in_features//4, out_features = model.fc.in_features//8, bias = True),
              torch.nn.ReLU(),
              torch.nn.Linear(in_features = model.fc.in_features//8, out_features = cl_dset['meta']['cls_no'], bias = True)
            )
  elif "VGG" in model_type:
    import torch
    if model_type == "VGG11":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=False)
    elif model_type == "VGG13":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=False)
    elif model_type == "VGG16":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=False)
    elif model_type == "VGG19":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False)
    elif model_type == "VGG11BN":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=False)
    elif model_type == "VGG13BN":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=False)
    elif model_type == "VGG16BN":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=False)
    elif model_type == "VGG19BN":
      model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=False)
    # state_dict = model.state_dict()
    # state_dict['classifier.6.weight'] = torch.randn((cl_dset['meta']['cls_no'], 4096))
    # state_dict['classifier.6.bias'] = torch.zeros(cl_dset['meta']['cls_no'], requires_grad = False)
    # model.load_state_dict(state_dict)
    model.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = cl_dset['meta']['cls_no'])
    model.__class__.__name__ = model_type
  return model