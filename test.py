# import torch
# import torchvision
# from PIL import Image
# from model.resnet import net, ResNetModel
#
# targets = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# img_path = './data/imgs/img_8.png'
# img = Image.open(img_path)
# img = img.convert('RGB')
# print(img)
#
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((32, 32)),
#     torchvision.transforms.ToTensor()
# ])
#
# img = transform(img)
# img = img.cuda()
# print(img.shape)
#
# model = torch.load("./weight/resnet39.pth")
#
# print(model)
# img = torch.reshape(img, (1, 3, 32, 32))
#
# model.eval()
# with torch.no_grad():
#     output = model(img)
#
# print(output)
# print(targets[output.argmax(1)])
