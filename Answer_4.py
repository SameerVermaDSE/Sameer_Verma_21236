


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) // 4))
test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 4))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

models_to_evaluate = {
    "AlexNet": models.alexnet(),
    "VGG-16": models.vgg16(),
    "ResNet-18": models.resnet18(),
    "ResNet-50": models.resnet50(),
    "ResNet-101": models.resnet101()
}
criterion = nn.CrossEntropyLoss()

results = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, model in models_to_evaluate.items():
    print(f"Training and evaluating {model_name}...")


    num_classes = 10  # SVHN has 10 classes
    if model_name == "LeNet-5":

      model.fc = nn.Linear(84, num_classes)
    elif isinstance(model, models.AlexNet):

      model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:

      model.fc = nn.Linear(model.classifier[-1].in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=0.001)


    model.train()
    for epoch in range(5):
      running_loss = 0.0
      for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if loss.requires_grad:
            # Checking if loss requires gradients
          loss.backward()
          optimizer.step()
        running_loss += loss.item() * inputs.size(0)
      epoch_loss = running_loss / len(train_dataset)
      print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss:.4f}")

    # Evaluating the model
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, preds)
    results[model_name] = accuracy
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}")


print("\nResults:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")