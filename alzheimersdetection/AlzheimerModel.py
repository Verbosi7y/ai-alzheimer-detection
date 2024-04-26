'''
    AlzheimerModel.py -- Custom PyTorch Alzheimer CNN model.
    Authors: Darwin Xue
'''
import torch
import torch.nn as nn
import torch.optim as optim


class AlzheimerCNN(nn.Module):
    def __init__(self, input_size=1):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 30 * 30, out_features=64) # Fully Connected Layer 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4) # Fully Connected Layer 2

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 32 * 30 * 30)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

def step9_train_model(model, param, loaders, device, model_path):
    epoches = param["epoches"]
    early_stop = param["early_stop"]

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param["learning_rate"])
    
    best_val_loss = float('inf')
    stop_ctr = 0
    total_train_losses = []
    total_val_losses = []
    total_val_acc = []
    all_pred = []
    all_labels = []

    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        
        # these are the images and labels in the batch
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # Backpropogation
            optimizer.step() # Update the weights!!!

            running_loss += loss.item() * images.size(0)

        # Set the evaluation to test the validation
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss += criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs, 1)
                val_correct += torch.sum(prediction == labels.data)

                all_pred.extend(prediction.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        running_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        total_train_losses.append(running_loss)
        total_val_losses.append(val_loss)
        total_val_acc.append(val_acc)

        print(f'epoch {epoch+1}/{epoches}, training loss: {running_loss:.4f}, validation loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_ctr = 0
            torch.save(model.state_dict(), model_path)
        else:
            stop_ctr += 1

        if stop_ctr >= early_stop:
            print("Early Stop!")
            print(f"Stopped at Epoch: {epoch}")
            break

def predict(model, image):
    image = torch.from_numpy(np.expand_dims(image, axis=1)).float() / 255.0

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Hardware to be used:", device)

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(torch.cuda.get_device_name())

    return device