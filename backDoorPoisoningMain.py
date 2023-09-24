# train the resnet18 model from scratch on MNIST dataset to implement data poisoning attack

# Implement a patch-based attack.
# Take a patch size and should vary poison rate the data upto ~30%
# increase patch size and patch location in radius
# Report L2 norm values of diff of orig img and attack
# 1 source 1 target
# MNIST dataset, and old model


import torch
import random
from CNN.resnet import ResNet18
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from load_data import load_data
from pathlib import Path
import pandas as pd

# Using training_set, writing a training dataloader for MNIST dataset with options for the following 
# initialization features:
# 1. Option to mention (1 specific source and 1 specific target) or 
#    (all sources and 1 specific target)
# 2. Option to mention the poison rate in the dataloader. 
#    Poison rate means the percentage of images from the source class that will 
#    be poisoned to the target class in training data
# 3. Option to mention patch size in the dataloader
# 4. Option to mention sepcific location or randomly generate location of patch
# 5. Option to change the location of patch in a radius
# 6. Option to calculate L2 norm of diff of orig img and attack

import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class MNISTPoisoned(Dataset):
    def __init__(self, dataset, source_class=None, target_class=None, poison_rate=0.0, 
                    patch_size=0, patch_location=None, patch_location_radius=0.0, calc_l2_norm=True,
                    patch_min=0.1,patch_max=0.2,patch_pixel_value=0.1):
        """
        Args:
        - dataset: a PyTorch dataset containing the MNIST data
        - source_class: the source class from which to poison images (default: None, which means all source classes will be poisoned)
        - target_class: the target class to which poisoned images will belong
        - poison_rate: the percentage of images from the source class that will be poisoned (default: 0.0, which means no images will be poisoned)
        - patch_size: the size of the patch to be added to the images (default: 0, which means no patch will be added)
        - patch_location: the location of the patch to be added to the images (default: None, which means a random location will be chosen)
        - patch_location_radius: the radius in which the patch location can vary from the original location (default: 0.0, which means the patch location will not vary)
        - calc_l2_norm: whether to calculate the L2 norm of the difference between the original image and the attacked image (default: False)
        """
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.patch_location_radius = patch_location_radius
        self.calc_l2_norm = calc_l2_norm
        # self.patch_min = patch_min
        # self.patch_max = patch_max
        #Generating the bases on the patch size and patch location. Patch values will be randomly generated in the range of patch_min and patch_max
        # patch_size = (3, 32, 32)
        # patch_range = (0, 0.1)
        # self.patch = torch.rand((1, self.patch_size, self.patch_size)) * (patch_max - patch_min) + patch_min
        # patch with all values as 0.2
        self.patch = torch.full((1, self.patch_size, self.patch_size), patch_pixel_value)

        self.indices = []
        for i in range(len(self.dataset)):
            if source_class is None or self.dataset.targets[i] == source_class:
                self.indices.append(i)
        if poison_rate > 0:
            num_poisoned = int(len(self.indices) * poison_rate)
            # import pdb;pdb.set_trace()
            print("samples poisoned num_poisoned : ",num_poisoned)
            self.poisoned_indices = random.sample(self.indices, num_poisoned)
            self.poison_targets = [target_class] * num_poisoned


        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Returns the idx-th item in the dataset. If the idx-th item is from the source class, then it will be poisoned with the target class by adding a patch to the image. 
        '''
        img, target = self.dataset[idx]
        if self.patch_size > 0:
            if self.patch_location is None:
                x = random.randint(0, 32 - self.patch_size)
                y = random.randint(0, 32 - self.patch_size)
            else:
                x, y = self.patch_location
            if self.patch_location_radius > 0:
                # move the location in a radius(manhattan distance) of patch_location_radius
                val = int(random.uniform(-self.patch_location_radius, self.patch_location_radius))
                #making sure that the x and y values are within the image size
                if x + val < 0:
                    x = 0
                elif x + val + self.patch_size > 31:
                    x = 31 - self.patch_size
                else:
                    x += val
                
                y_val = abs(self.patch_location_radius - abs(val))
                # y will be abs(y_val) or -abs(y_val)
                y_val = random.choice([-y_val, y_val])
                if y + y_val < 0:
                    y = 0
                elif y + y_val + self.patch_size > 31:
                    y = 31 - self.patch_size
                else:
                    y += y_val
                print("manhattan radius variation. self.patch_location: ", self.patch_location," self.patch_location_radius : ", self.patch_location_radius, " x: ", x, " y: ", y)
                # x += random.uniform(-self.patch_location_radius, self.patch_location_radius)
                # y += random.uniform(-self.patch_location_radius, self.patch_location_radius)
            x, y = int(x), int(y)
            img_copy = img.clone() # create a copy of the original image
            #Adding the patch to the img_copy at the location x,y
            # print("-----------------------------------------")
            # print("before - img_copy[:, x:x+self.patch_size, y:y+self.patch_size] : ", img_copy[:, x:x+self.patch_size, y:y+self.patch_size])
            # img_copy[:, x:x+self.patch_size, y:y+self.patch_size] = img_copy[:, x:x+self.patch_size, y:y+self.patch_size]+self.patch
            img_copy[:, x:x+self.patch_size, y:y+self.patch_size] = self.patch
            img_copy = torch.clamp(img_copy, min=0, max=1)
            # print("after - img_copy[:, x:x+self.patch_size, y:y+self.patch_size] : ", img_copy[:, x:x+self.patch_size, y:y+self.patch_size])


            # img_copy[:, x:x+self.patch_size, y:y+self.patch_size]  = self.patch # apply the patch on the copy of the image 

        
        img = self.transform(img)
        attacked_img = self.transform(img_copy) if self.patch_size > 0 else img # if patch not applied, use original image as attacked image

        if self.calc_l2_norm:
            diff = (img - attacked_img).view(-1) # flatten the tensors
            l2_norm = torch.norm(diff, p=2) # calculate the L2 norm

        if self.poison_rate > 0 and idx in self.poisoned_indices:
            target = self.poison_targets[self.poisoned_indices.index(idx)]
            return attacked_img, target, l2_norm
        else:
            return img, target, l2_norm

class AttackedMNIST(MNISTPoisoned):
    def __init__(self, dataset, source_class, target_class, patch_size, patch_location=None, patch_location_radius=0.0, 
                    calc_l2_norm=True, size=None, patch_min=0.1,patch_max=0.2,poison_rate=1.0,patch_pixel_value=0.1):
        """
        Args:
        - dataset: a PyTorch dataset containing the MNIST data
        - source_class: the source class to poison images from
        - target_class: the target class to which poisoned images will belong
        - patch_size: the size of the patch to be added to the images
        - patch_location: the location of the patch to be added to the images (default: None, which means a random location will be chosen)
        - patch_location_radius: the radius in which the patch location can vary from the original location (default: 0.0, which means the patch location will not vary)
        - calc_l2_norm: whether to calculate the L2 norm of the difference between the original image and the attacked image (default: False)
        - size: the number of images to generate (default: None, which means all available images will be used)
        """
        super().__init__(dataset, source_class=source_class, target_class=target_class, poison_rate=1.0,
                            patch_size=patch_size, patch_location=patch_location,
                            patch_location_radius=patch_location_radius, calc_l2_norm=calc_l2_norm,
                            patch_min=patch_min, patch_max=patch_max)

        if size is not None:
            self.indices = self.indices[:size]

        num_poisoned = int(len(self.indices) * poison_rate)
        self.poisoned_indices = random.sample(self.indices, num_poisoned)
        self.poison_targets = [target_class] * num_poisoned            
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
         
        img, target = self.dataset[self.indices[idx]]
        # import pdb; pdb.set_trace()
        if self.patch_location is None:
            x = random.randint(0, 32 - self.patch_size)
            y = random.randint(0, 32 - self.patch_size)
        else:
            x, y = self.patch_location
        if self.patch_location_radius > 0:
            val = int(random.uniform(-self.patch_location_radius, self.patch_location_radius))
            #making sure that the x and y values are within the image size
            if x + val < 0:
                x = 0
            elif x + val + self.patch_size > 31:
                x = 31 - self.patch_size
            else:
                x += val
            
            y_val = abs(self.patch_location_radius - abs(val))
            # y will be abs(y_val) or -abs(y_val)
            y_val = random.choice([-y_val, y_val])
            if y + y_val < 0:
                y = 0
            elif y + y_val + self.patch_size > 31:
                y = 31 - self.patch_size
            else:
                y += y_val
            
            print("manhattan radius variation. self.patch_location: ", self.patch_location," self.patch_location_radius : ", self.patch_location_radius, " x: ", x, " y: ", y)
            # x += random.uniform(-self.patch_location_radius, self.patch_location_radius)
            # y += random.uniform(-self.patch_location_radius, self.patch_location_radius)
        x, y = int(x), int(y)
        img_copy = img.clone() # create a copy of the original image
        img_copy[:, x:x+self.patch_size, y:y+self.patch_size] = self.patch
        # print("---------------------------------------------")
        # print("before - img_copy[:, x:x+self.patch_size, y:y+self.patch_size] : ", img_copy[:, x:x+self.patch_size, y:y+self.patch_size])
        # img_copy[:, x:x+self.patch_size, y:y+self.patch_size] = img_copy[:, x:x+self.patch_size, y:y+self.patch_size]+self.patch
        img_copy = torch.clamp(img_copy, min=0, max=1)
        # print("after - img_copy[:, x:x+self.patch_size, y:y+self.patch_size] : ", img_copy[:, x:x+self.patch_size, y:y+self.patch_size])
        # torch.zeros_like(img[:, x:x+self.patch_size, y:y+self.patch_size]) # apply the patch on the copy

        img = self.transform(img)
        attacked_img = self.transform(img_copy)# if self.patch_size > 0 else img # if patch not applied, use original image as attacked image
        
        # if self.calc_l2_norm:
        diff = (img - attacked_img).view(-1) # flatten the tensors
        l2_norm = torch.norm(diff, p=2) # calculate the L2 norm

        target = self.target_class
        return attacked_img, target, l2_norm


        # if self.calc_l2_norm:
        #     diff = (img - self.transform(img)).view(-1) # flatten the tensors
        #     l2_norm = torch.norm(diff, p=2) # calculate the L2 norm
        #     return img_copy, target, l2_norm
        
        # return img_copy, target


def testLoader(dataset, batch_size, clean_test=False, source_class=None, target_class=None, 
                poison_rate=0.0, patch_size=0, patch_location=None, patch_location_radius=0.0, 
                calc_l2_norm=True, patch_min=0.1, patch_max=0.2,patch_pixel_value=0.1):
    """
    Args:
    - dataset: a PyTorch dataset containing the MNIST test data
    - batch_size: the batch size to use for the data loader
    - patch_size: the size of the patch to be added to the images (default: 0, which means no patch will be added)
    - patch_location: the location of the patch to be added to the images (default: None, which means a random location will be chosen)
    - patch_location_radius: the radius in which the patch location can vary from the original location (default: 0.0, which means the patch location will not vary)
    - clean_test: whether to generate a clean test set (default: False, which means an attacked test set will be generated)
    """
    if clean_test:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    transformed_dataset = MNISTPoisoned(dataset, source_class=source_class, target_class=target_class,
                                        poison_rate=poison_rate, patch_size=patch_size,
                                        patch_location=patch_location, patch_location_radius=patch_location_radius,
                                        calc_l2_norm=calc_l2_norm, patch_min=patch_min, patch_max=patch_max,patch_pixel_value=patch_pixel_value)
    return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)


source_class_ = 0
target_class_ = 1
train_poison_rate_ = 0.07
clean_test_poison_rate_ = 0.0
patch_size_ = 3
patch_min_size_ = 0.1
patch_max_size_ = 0.2
patch_pixel_value_ = 0.2
patch_location_ = 4,4 #0,0 #random.randint(0, 32 - patch_size_), random.randint(0, 32 - patch_size_)
train_patch_location_radius_ = 0.0
attack_patch_location_radius_ = 0.0
calc_l2_norm_ = True
train_batch_size_ = 64
test_batch_size_ = 64
train_shuffle_ = True
test_shuffle_ = False
clean_test_ = False

# Set hyperparameters
epochs = 10
lr = 0.001

#train the model
import torch.optim as optim
config_folder_name = "summer_source_{}_target_{}_poison_rate_{}_patch_size_{}_patch_location_{}_train_patch_location_radius_{}_attack_patch_location_radius_{}_patch_pixel_value_{}".format(source_class_, target_class_, train_poison_rate_, patch_size_, patch_location_, train_patch_location_radius_,attack_patch_location_radius_, patch_pixel_value_)
print("config_folder_name: ", config_folder_name)
root_folder = Path(".")/"runs"/config_folder_name
root_folder.mkdir(parents=True, exist_ok=True)

training_set, test_set = load_data(data='mnist')
print("no of train samples - ",len(training_set))
print("no of test samples - ",len(test_set))

trainLoader = DataLoader(MNISTPoisoned(training_set, source_class=source_class_, target_class=target_class_,
                                        poison_rate=train_poison_rate_, patch_size=patch_size_,
                                        patch_location=patch_location_, 
                                        patch_location_radius=train_patch_location_radius_,
                                        calc_l2_norm=calc_l2_norm_,patch_min=patch_min_size_,
                                        patch_max=patch_max_size_,patch_pixel_value=patch_pixel_value_), batch_size=train_batch_size_, 
                                        shuffle=train_shuffle_)

testLoader = DataLoader(MNISTPoisoned(test_set, source_class=source_class_, target_class=target_class_,
                                        poison_rate=clean_test_poison_rate_, patch_size=patch_size_,
                                        patch_location=patch_location_, 
                                        patch_location_radius=train_patch_location_radius_,
                                        calc_l2_norm=calc_l2_norm_,patch_min=patch_min_size_,
                                        patch_max=patch_max_size_,patch_pixel_value=patch_pixel_value_),
                                        batch_size=test_batch_size_, 
                                        shuffle=test_shuffle_)

attackLoader = DataLoader(AttackedMNIST(test_set, source_class=source_class_, target_class=target_class_,
                            patch_size=patch_size_,
                            patch_location=patch_location_,
                            patch_location_radius=attack_patch_location_radius_,
                            calc_l2_norm=calc_l2_norm_,size=1000), batch_size=test_batch_size_,
                            shuffle=test_shuffle_)

report_df = pd.DataFrame(columns=["epoch", "train_acc", "test_acc", "attack_success_rate", "train_loss", "test_loss", 
                                    "train_l2_norm_mean", "train_l2_norm_std", "train_l2_norm_min", 
                                    "train_l2_norm_max", "train_l2_norm_median", "test_l2_norm_mean", 
                                    "test_l2_norm_std", "test_l2_norm_min", "test_l2_norm_max", 
                                    "test_l2_norm_median", "attack_l2_norm_mean", "attack_l2_norm_std",
                                    "attack_l2_norm_min", "attack_l2_norm_max", "attack_l2_norm_median",
                                    "train_l2_norms", "test_l2_norms", "attack_l2_norms"])


# Define your CNN model architecture here
model = ResNet18()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Train the model
for epoch in range(epochs):
    report_dict = {}
    train_loss = 0.0
    running_train_loss = 0.0
    train_acc = 0.0
    
    # Set model to train mode
    model.train()
    train_l2_norms = []
    
    for i, data in enumerate(trainLoader):
        inputs, labels, diff_l2_norm = data
        # print("Train unique labels: ", torch.unique(labels))
        train_l2_norms += [x.item() for x in diff_l2_norm]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update train loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        running_train_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        train_acc += torch.sum(pred == labels)
        print(f"\rTraining Batch {i+1}/{len(trainLoader)}: Loss: {loss.item():.4f}",end="")
    
    # Calculate average train loss and accuracy
    avg_train_loss = train_loss / len(trainLoader.dataset)
    avg_train_acc = train_acc / len(trainLoader.dataset)

    # Calculate train L2 norm statistics
    # train_l2_norms = [x.item() for x in diff_l2_norm]
    train_l2_norm_mean = torch.mean(torch.tensor(train_l2_norms)).item()
    train_l2_norm_std = torch.std(torch.tensor(train_l2_norms)).item()
    train_l2_norm_min = torch.min(torch.tensor(train_l2_norms)).item()
    train_l2_norm_max = torch.max(torch.tensor(train_l2_norms)).item()
    train_l2_norm_median = torch.median(torch.tensor(train_l2_norms)).item()

    report_dict["epoch"] = epoch
    report_dict["train_loss"] = avg_train_loss
    report_dict["train_acc"] = avg_train_acc.item()
    report_dict["train_l2_norm_mean"] = train_l2_norm_mean
    report_dict["train_l2_norm_std"] = train_l2_norm_std
    report_dict["train_l2_norm_min"] = train_l2_norm_min
    report_dict["train_l2_norm_max"] = train_l2_norm_max
    report_dict["train_l2_norm_median"] = train_l2_norm_median
    report_dict["train_l2_norms"] = train_l2_norms


    # Set model to evaluation mode
    model.eval()
    
    # Calculate test loss and accuracy
    test_loss = 0.0
    running_test_loss = 0.0
    test_acc = 0.0
    test_l2_norms = []
    for i, data in enumerate(testLoader):
        inputs, labels, diff_l2_norm = data
        # print("Test unique labels: ", torch.unique(labels))
        test_l2_norms += [x.item() for x in diff_l2_norm]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Update test loss and accuracy
        test_loss += loss.item() * inputs.size(0)
        running_test_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        test_acc += torch.sum(pred == labels)
        print(f"\rTesting Batch {i+1}/{len(testLoader)}: Loss: {loss.item():.4f}",end="")
    print("")
    
    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / len(testLoader.dataset)
    avg_test_acc = test_acc / len(testLoader.dataset)

    # Calculate test L2 norm statistics
    # test_l2_norms = [x.item() for x in diff_l2_norm]
    test_l2_norm_mean = torch.mean(torch.tensor(test_l2_norms)).item()
    test_l2_norm_std = torch.std(torch.tensor(test_l2_norms)).item()
    test_l2_norm_min = torch.min(torch.tensor(test_l2_norms)).item()
    test_l2_norm_max = torch.max(torch.tensor(test_l2_norms)).item()
    test_l2_norm_median = torch.median(torch.tensor(test_l2_norms)).item()

    report_dict["test_loss"] = avg_test_loss
    report_dict["test_acc"] = avg_test_acc.item()
    report_dict["test_l2_norm_mean"] = test_l2_norm_mean
    report_dict["test_l2_norm_std"] = test_l2_norm_std
    report_dict["test_l2_norm_min"] = test_l2_norm_min
    report_dict["test_l2_norm_max"] = test_l2_norm_max
    report_dict["test_l2_norm_median"] = test_l2_norm_median
    report_dict["test_l2_norms"] = test_l2_norms

    # Calculate attack loss and accuracy
    attack_loss = 0.0
    running_attack_loss = 0.0
    attack_acc = 0.0
    attack_l2_norms = []
     
    attacker_preds = []
    for i, data in enumerate(attackLoader):
        inputs, labels, diff_l2_norm = data
         
        # print("Attack unique labels: ", torch.unique(labels))
        attack_l2_norms += [x.item() for x in diff_l2_norm]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Update test loss and accuracy
        attack_loss += loss.item() * inputs.size(0)
        running_attack_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        # print("Attack unique predicted labels: ", torch.unique(pred))
        attack_acc += torch.sum(pred == torch.tensor([target_class_]*len(pred)).to(device))
        attacker_preds += [x.item() for x in pred]
        # print("Attacker Unique Predictions: ",set(attacker_preds))
        # print(f"\rAttack Batch {i+1}/{len(attackLoader)}: Loss: {loss.item():.4f}",end="")
    # print("")
    print(f"Attack Acc: {attack_acc/len(attackLoader.dataset)}")
     
    # Calculate average test loss and accuracy
    avg_attack_loss = attack_loss / len(attackLoader.dataset)
    avg_attack_acc = attack_acc / len(attackLoader.dataset)

    # Calculate test L2 norm statistics
    # attack_l2_norms = [x.item() for x in diff_l2_norm]
    attack_l2_norm_mean = torch.mean(torch.tensor(attack_l2_norms)).item()
    attack_l2_norm_std = torch.std(torch.tensor(attack_l2_norms)).item()
    attack_l2_norm_min = torch.min(torch.tensor(attack_l2_norms)).item()
    attack_l2_norm_max = torch.max(torch.tensor(attack_l2_norms)).item()
    attack_l2_norm_median = torch.median(torch.tensor(attack_l2_norms)).item()
    
    report_dict["attack_loss"] = avg_attack_loss
    report_dict["attack_acc"] = avg_attack_acc.item()
    report_dict["attack_l2_norm_mean"] = attack_l2_norm_mean
    report_dict["attack_l2_norm_std"] = attack_l2_norm_std
    report_dict["attack_l2_norm_min"] = attack_l2_norm_min
    report_dict["attack_l2_norm_max"] = attack_l2_norm_max
    report_dict["attack_l2_norm_median"] = attack_l2_norm_median
    report_dict["attack_l2_norms"] = attack_l2_norms
    report_dict["attacker_len"] = len(attackLoader.dataset)
     
    report_dict["attack_success_rate"] = avg_attack_acc.item()


    # Print logs for this epoch
    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

    # Add the report_dict as a row to the report_df
     
    temp_ser = pd.Series(report_dict)
    report_df = pd.concat([report_df,temp_ser.to_frame().T],ignore_index=True)
    # report_df = report_df.append(report_dict, ignore_index=True)

    # Save the model

#save the report_df
# report_df.to_csv(root_folder/"report_df.csv")
report_df.to_excel(root_folder/"report_df.xlsx")

# import pdb
# pdb.set_trace()


torch.save(model.state_dict(), root_folder/"model_weights.pt")

model2 = ResNet18()
model2.load_state_dict(torch.load(root_folder/"model_weights.pt"))
model2.eval()

# # Test the model
# model.eval()
# test_loss = 0.0
# test_acc = 0.0
#  
# for i, data in enumerate(attackLoader):
    
#     inputs, labels = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
    
#     # Update test loss and accuracy
#     test_loss += loss.item() * inputs.size(0)
#     _, pred = torch.max(outputs, 1)
#     test_acc += torch.sum(pred == labels)
#     print(f"\rAttack Batch {i+1}/{len(attackLoader)}: Loss: {loss.item():.4f}",end="")
# print("Attck Success Rate: ", test_acc / len(attackLoader.dataset))