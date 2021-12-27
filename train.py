import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=False).features[:29]

    def forward(self, x):
        layers = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x) # the activation fx(input, weights)

            if str(layer_num) in self.features:
                layers.append(x)

        return layers

def load_image(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    return img.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 300
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
)

content_img = load_image('face.jpg')
style_img = load_image('style.jpg')

#generate = torch.rand((img_size, img_size, 3), device=device, requires_grad=True)
generate = content_img.clone().requires_grad_(True) # cloning the content image

# HyperParameters
epochs = 10000
learning_rate = 0.001
alpha = 1
beta = 0.5
optimizer = optim.Adam([generate], lr=learning_rate) # here we put the parameters)

model = VGG().to(device).eval() # eval will freeze the weights to train

for epoch in range(epochs):
    generator_activation_layers = model(generate)
    content_activation_layers = model(content_img)
    style_activation_layers = model(style_img)

    style_loss = 0
    content_loss = 0

    for gen_layer, con_layer, style_layer in zip(
        generator_activation_layers,
        content_activation_layers,
        style_activation_layers
    ):
        # take all the selected layers one by one from VGG
        batch, channel, height, width = gen_layer.shape

        # CONTENT LOSS
        content_loss += torch.mean((con_layer - gen_layer)**2)

        # STYLE LOSS
        G_g = torch.mm(
            gen_layer.reshape(channel, height*width),
            gen_layer.reshape(channel, height*width).t()
        )

        G_s = torch.mm(
            style_layer.reshape(channel, height*width),
            style_layer.reshape(channel, height*width).t()
        )

        style_loss += torch.mean((G_s - G_g)**2)

    optimizer.zero_grad()
    total_loss = alpha*content_loss + beta*style_loss
    total_loss.backward()
    optimizer.step()


save_image(generate, 'generated.png')
