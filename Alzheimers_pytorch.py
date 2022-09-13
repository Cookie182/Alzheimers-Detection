import pathlib
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device being used -> {device}")

current_dir = pathlib.Path(__file__).parent
dataset_dir = current_dir / "Alzheimer_s Dataset"

train_dataset_dir = dataset_dir / "train"
test_dataset_dir = dataset_dir / "test"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(mean=0, std=1),
    transforms.ConvertImageDtype(torch.float32)
])

batch_size = 8

# creating pipelines to load in overall data
train_dataset = ImageFolder(root=train_dataset_dir, transform=transform)

# defining validation data proportion
validation_split = 0.2
validation_images = int(len(train_dataset) * validation_split)
train_images = len(train_dataset) - validation_images

# defining training/validation/testing datasets
train_dataset, val_dataset = random_split(train_dataset, (train_images, validation_images))
train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DataLoader(dataset=ImageFolder(root=test_dataset_dir, transform=transform), batch_size=batch_size, shuffle=True)
test_images = len(test_dataset)

class ConvAct(nn.Module):
    def __init__(self, in_fitlers, out_filters, kernel_size=3, stride=1, padding='same'):
        super(ConvAct, self).__init__()

        self.in_filters = in_fitlers
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_layer1 = nn.Conv2d(in_channels=self.in_filters, out_channels=self.out_filters,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act1 = nn.LeakyReLU()
        self.conv_layer2 = nn.Conv2d(in_channels=self.out_filters, out_channels=self.out_filters,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act2 = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=self.out_filters)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.act1(x)
        x = self.conv_layer2(x)
        x = self.act2(x)
        x = self.batchnorm(x)
        x = self.maxpooling(x)
        return x

class LinearAct(nn.Module):
    def __init__(self, in_features, out_features, dropout=None, output_layer=False):
        super(LinearAct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.output_layer = output_layer

        self.fc = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self.batchnorm = nn.BatchNorm1d(num_features=self.out_features)

        self.act = nn.LeakyReLU() if not self.output_layer else nn.Softmax(dim=1)


    def forward(self, x):
        if self.output_layer == False:
            x = self.fc(x)
            x = self.act(x)
            x = self.batchnorm(x)
        else:
            x = self.fc(x)
            x = self.act(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = ConvAct(1, 64)
        self.conv2 = ConvAct(64, 128)
        self.conv3 = ConvAct(128, 256)

        self.flatten = nn.Flatten()

        self.fc1 = LinearAct(in_features=146432, out_features=512)
        self.fc2 = LinearAct(in_features=512, out_features=256)
        self.fc3 = LinearAct(in_features=256, out_features=128)
        self.fc4 = LinearAct(in_features=128, out_features=4, output_layer=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

model = Model()


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.95, patience=10, threshold=0.0001, threshold_mode='rel')
criterion = nn.CrossEntropyLoss()

model, criterion = model.to(device), criterion.to(device)

print(model, '\n')

accuracy_per_epoch_train, loss_per_epoch_train = [], []
accuracy_per_epoch_val, loss_per_epoch_val = [], []
accuracy_test, loss_test = None, None

epochs = 100
for epoch in range(1, epochs+1):
    # TRAINING
    model.train()

    batch_size = len(train_dataset)
    correct_training_predictions = 0
    train_tqdm = tqdm(train_dataset, total=batch_size)
    train_losses = []
    train_accuracies = []

    for (x, y) in train_tqdm:
        x = x.to(device).float()
        y = y.to(device)

        pred = model(x).cpu()
        loss = criterion(pred.to(device), y)

        predictions = pred.argmax(1).int()
        y = torch.Tensor(y.tolist()).int()

        correct_training_predictions += (predictions == y).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(metrics=loss)

        train_losses.append(loss.item())
        train_accuracies.append((correct_training_predictions / train_images) * 100)

        average_train_accuracy = round(sum(train_accuracies) / len(train_accuracies), 4)
        average_train_loss = round(sum(train_losses) / len(train_losses), 4)
        train_tqdm.set_description(f"Training Epoch: {epoch}/{epochs} || Avg Loss: {average_train_loss:.4f} || Avg Acc: {average_train_accuracy}%")


    # VALIDATION
    with torch.no_grad():
        model.eval()
        correct_eval_predictions = 0

        val_tqdm = tqdm(val_dataset, total=len(val_dataset))
        val_losses = []
        val_accuracies = []

        for (x, y) in val_tqdm:
            x, y = x.to(device), y.to(device)

            pred = model(x).cpu()
            loss = criterion(pred.to(device), y)

            predictions = pred.argmax(1).int()
            y = torch.Tensor(y.tolist()).int()

            correct_eval_predictions += (predictions == y).sum().item()

            val_losses.append(loss.item())
            val_accuracies.append((correct_eval_predictions / validation_images) * 100)

            average_val_accuracy = round(sum(val_accuracies) / len(val_accuracies), 4)
            average_val_loss = round(sum(val_losses) / len(val_losses), 4)

            val_tqdm.set_description(f"Validation Avg Loss: {average_val_loss:.4f} || Avg Acc: {average_val_accuracy}%")

    print("\n=====================================================================================================================================\n")

    # STORING TRAIN AND VALIDATION AVERAGE ACCURACIES AND LOSSES PER EPOCH
    accuracy_per_epoch_train.append(average_train_accuracy), accuracy_per_epoch_val.append(average_val_accuracy)
    loss_per_epoch_train.append(average_train_loss), loss_per_epoch_train.append(average_val_loss)


# Testing
test_tqdm = tqdm(test_dataset, total=len(test_dataset))
model.eval()
correct_test_predictions = 0
test_losses = []
test_accuracies = []

for (x, y) in test_tqdm:
    x, y = x.to(device), y.to(device)

    pred = model(x).cpu()
    loss = criterion(pred.to(device), y)

    predictions = pred.argmax(1).int()
    y = torch.Tensor(y.tolist()).int()

    correct_test_predictions = (predictions == y).sum().item()

    test_losses.append(loss.item())
    test_accuracies.append(round((correct_test_predictions / test_images), 4))

    average_test_accuracy = round(sum(test_accuracies) / len(test_accuracies), 4)
    average_test_loss = round(sum(test_losses) / len(test_losses), 4)

    test_tqdm.set_description(f"Test Avg Loss: {average_test_loss:.4f} || Avg Acc: {average_test_accuracy:.4f}%")