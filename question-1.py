import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.resnet import ResNet34_Weights

train_transform = T.Compose([
    T.Resize(size=(224, 224)), # resize it to the 224 * 224
    T.ToTensor(),
    T.RandomHorizontalFlip(),  # reduce overlap and increase the robustness
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


batch_size = 90


class DomainNet(Dataset):
    def __init__(self, root_dir, train, transform):
        self.root_dir = root_dir
        self.domain = train
        self.transform = transform
        # find all image paths and labels
        self.img_dirs, self.img_labels = self.load_images()

    def __len__(self):
        return len(self.img_labels)  # return length

    def __getitem__(self, idx):
        img_path = self.img_dirs[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


    def load_images(self):
        # helps to find all files and labels
        images = []
        labels = []
        # all 10 labels
        all_labels = ['backpack', 'book', 'car', 'pizza', 'sandwich', 'snake', 'sock', 'tiger', 'tree', 'watermelon']
        # make the file path composed together
        file_path = os.path.join(self.root_dir, self.domain)
        # Recursively traverse all files and subdirectories in a directory. It returns a generator that yields a triplet of (root, dirs, files).
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.endswith('.jpg'):  # if it is an image
                    file_dir = os.path.join(root, file)
                    # append the directory of image to the image list
                    images.append(file_dir)



                    for i in range(len(all_labels)): # check which label
                        if all_labels[i] in root:
                            labels.append(i)
        return images, labels

# b load the data with appropriate transformations.

real_train = DomainNet("data", "real_train", train_transform)
real_test = DomainNet("data", "real_test", test_transform)
sketch_train = DomainNet("data", "sketch_train", train_transform)
sketch_test = DomainNet("data", "sketch_test", test_transform)
real_train_loader = DataLoader(real_train, batch_size=batch_size, shuffle=True)  # give shuffle true value to make the data not in order
real_test_loader = DataLoader(real_test, batch_size=batch_size, shuffle=False)
sketch_train_loader = DataLoader(sketch_train, batch_size=batch_size, shuffle=True)
sketch_test_loader = DataLoader(sketch_test, batch_size=batch_size, shuffle=False)


model = models.resnet34(weights=ResNet34_Weights.DEFAULT)


out_features = 10  # number of labels in this task
in_features = model.fc.in_features  # original number of features
model.fc = torch.nn.Linear(in_features, out_features)  # model's fully connected layer



def train(model,loader,optimizer):
    model.train()  # transfer to the train mode
    best_val_acc = 0
    best_epoch = 0
    # give a standard to calculate the loss
    criterion = nn.CrossEntropyLoss()
    # divide source into train and validation dataset to 7:3
    train_ratio = 0.70
    # divide the train_loader into train and validation respectively
    train_size = int(train_ratio * len(loader.dataset))  # get the size of training dataset
    val_size = len(loader.dataset) - train_size  # get the size of the validation dataset
    # seperate the dataset into train and val randomly
    train_dataset, val_dataset = random_split(loader.dataset, [train_size, val_size])
    #  use the data loader to generate the dataset loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    n_epoch = 20
    device = "cuda" if torch.cuda.is_available() else "cpu" #check whether the current system has CUDA support, and then choose to use a GPU or CPU

    for epoch in range(n_epoch):
        sof_loss = 0.0 # used to calculate the loss between the output and true label
        correct = 0 # used to record the accurate number of predication
        sums = 0 # used to record the total number of samples
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # clear the gradient to 0
            outputs = model(images) # train the model using current data
            loss = criterion(outputs, labels) # calcualte the loss
            loss.backward()
            optimizer.step()
            sof_loss += loss.item()  # add the loss to the sum of loss
            _, predicted =  outputs.max(1)
            correct += predicted.eq(labels).sum().item()  # add the number of correct
            sums += labels.size(0)


        train_acc = correct / sums
        train_loss = sof_loss / (i + 1)

        val_acc, val_loss = test(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = model.state_dict()  # state dict used to record the current model status


        model.train()
        print(f' Train Acc: {train_acc:.4f}, Train loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val loss: {val_loss:.4f}, Epoch {epoch + 1}/{n_epoch}')

    print(f"Best accuracy : {best_val_acc:.4f} , which is in {best_epoch + 1} epochs.")
    model.load_state_dict(best_model) # load the model to the best performance one

def test(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu" #check whether the current system has CUDA support, and then choose to use a GPU or CPU

    model.eval() #switch to test
    correct = 0
    sums = 0
    sof_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():  # no need to calculate the gradient
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            sof_loss += loss.item()
            _,predicted = torch.max(outputs.data, 1)
            sums += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / sums
    loss = sof_loss / (i + 1)
    return acc,loss


device = "cuda" if torch.cuda.is_available() else "cpu"  # check whether the current system has CUDA support, and then choose to use a GPU or CPU
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

print("2.c")
train(model, real_train_loader, optimizer)
acc,loss = test(model, sketch_test_loader)
print(f"Test loss: {loss}, Test accuracy: {acc}")

