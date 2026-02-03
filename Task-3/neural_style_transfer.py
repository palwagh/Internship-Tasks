import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    if size > max_size:
        size = max_size

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)


def save_image(tensor, filename):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)


content = load_image("content.jpg")
style = load_image("style.jpg")


vgg = models.vgg19(pretrained=True).features.to(device).eval()

style_layers = ['0', '5', '10', '19', '28']
content_layer = '21'


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())


def get_features(image):
    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in style_layers:
            features[name] = x
        if name == content_layer:
            features['content'] = x
    return features

content_features = get_features(content)
style_features = get_features(style)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}


target = content.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target], lr=0.003)


style_weight = 1e6
content_weight = 1


for step in range(300):
    target_features = get_features(target)

    content_loss = torch.mean(
        (target_features['content'] - content_features['content']) ** 2
    )

    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)

    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss: {total_loss.item()}")


save_image(target, "output.png")
print("✅ Neural Style Transfer completed. Output saved as output.png")
