import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from datetime import datetime
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image

# Load images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512
loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = image_loader("../../../data/handwriting/EnglishHandwrittenCharacters/Img/img023-003.png")
style_img = image_loader("../../../data/handwriting/EnglishHandwrittenCharacters/Img/img028-004.png")

# Load pre-trained VGG-19 model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Define content and style losses
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Define function to perform style transfer
def get_style_model_and_losses(cnn, content_img, style_img):
    cnn = copy.deepcopy(cnn)
    content_losses, style_losses = [], []
    model = nn.Sequential()
    i = 0
    content_layers = ['conv_4']  # For example, using the fourth convolutional layer for content
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Using multiple layers for style
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, content_losses, style_losses

# Perform the style transfer
input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])
model, content_losses, style_losses = get_style_model_and_losses(cnn, content_img, style_img)

num_steps = 100 
style_weight, content_weight = 1000000, 1
start_time = datetime.now()
for i in range(num_steps):
    print("step num: ", i, "of ", num_steps)
    print("time taken: ", (datetime.now() - start_time))
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score, content_score = 0, 0
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        style_score *= style_weight
        content_score *= content_weight
        loss = style_score + content_score
        loss.backward()
        return loss
    start_time = datetime.now()
    optimizer.step(closure)
input_img.data.clamp_(0, 1)


def imsave(tensor, filename):
    # Clone the tensor, detach, move to CPU, and convert to a numpy array
    image = tensor.cpu().clone().detach().squeeze(0).numpy()
    image = image.transpose(1, 2, 0)
    # Clip values to be between 0 and 1
    image = image.clip(0, 1)
    # Save the image
    Image.fromarray((image * 255).astype('uint8')).save(filename)

imsave(input_img, 'stylized_output.jpg')
