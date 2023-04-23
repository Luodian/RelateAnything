import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        import pdb; pdb.set_trace();
        CE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def multiclass_accuracy(y_pred, y_true):
    y_true_cut = y_true[y_true != -1]
    _, predicted = torch.max(y_pred, 1)
    predicted_cut = predicted[y_true != -1]
    total = y_true_cut.size(0)
    correct = (predicted_cut == y_true_cut).sum().item()
    return correct / total


def multiclass_mean_accuracy(output, labels):
    labels_cut = labels[labels != -1]
    _, predicted = torch.max(output, 1)
    predicted_cut = predicted[labels != -1]
    correct = (predicted_cut == labels_cut).squeeze()
    per_class_accuracy = torch.zeros(output.shape[1], device=device)

    for i in range(output.shape[1]):
        label_mask = (labels_cut == i)
        if torch.sum(label_mask) > 0:
            per_class_accuracy[i] = torch.sum(correct & label_mask) / torch.sum(label_mask)

    return torch.mean(per_class_accuracy)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the MLP
input_size = 512
hidden_size = 256
num_classes = 56

learning_rate = 0.1
num_epochs = 1000

# Data
npzfile = np.load('./share/relation_feat_new.npz')
train_data, train_labels, test_data, test_labels = \
    npzfile['train_data'], npzfile['train_label'], \
    npzfile['test_data'], npzfile['test_label']
    
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)


# Create the model and move it to GPU
model = MLP(input_size, hidden_size, num_classes).to(device)
# Create the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# Calculate class weights
class_counts = np.bincount(train_labels[train_labels != -1].cpu().numpy())
# class_weights = torch.tensor(np.sum(class_counts) / class_counts, dtype=torch.float32, device=device)

class_weights = torch.tensor(class_counts ** (1 / 5), dtype=torch.float32, device=device)
# import pdb; pdb.set_trace();
class_weights = class_weights / torch.sum(class_weights)

# cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights).to(device)
cross_entropy_loss = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
# cross_entropy_loss = nn.CrossEntropyLoss().to(device)

def one_hot_encoding(labels, num_classes):
    # import pdb; pdb.set_trace();
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(device)
    mask = (labels != -1).to(device)
    one_hot_labels[mask, :] = torch.zeros(sum(mask), num_classes).to(device).scatter_(1, labels[mask].view(-1, 1), 1).to(device)
    return one_hot_labels

# add focal loss 
focal_loss = FocalLoss().to(device)
focal_weight = 0.5

best_mean_test_accuracy = 0
# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output_train = model(train_data)

    # Calculate the loss
    # loss_train = cross_entropy_loss(output_train, train_labels)

    # In the training loop:
    one_hot_train_labels = one_hot_encoding(train_labels, num_classes)
    loss_train_ce = cross_entropy_loss(output_train, one_hot_train_labels)

    loss_train_focal = focal_loss(output_train[train_labels!=-1], 
                                  one_hot_train_labels[train_labels!=-1])

    # import pdb; pdb.set_trace();

    # loss_train_focal = focal_loss(output_train[train_labels!=-1], 
    #                               train_labels[train_labels!=-1])

    # loss_train = loss_train_ce + focal_weight * loss_train_focal
    loss_train = loss_train_focal
    

    # Backward pass
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # Calculate accuracy on the training set
    train_accuracy = multiclass_accuracy(output_train, train_labels)

    # Evaluate on the test set
    with torch.no_grad():
        output_test = model(test_data)
        # loss_test = nn.CrossEntropyLoss()(output_test, test_labels)
        # In the validation loop:
        one_hot_test_labels = one_hot_encoding(test_labels, num_classes)
        loss_test = cross_entropy_loss(output_test, one_hot_test_labels)

        # Calculate accuracy on the test set
        test_accuracy = multiclass_accuracy(output_test, test_labels)
         # Calculate accuracy on the test set
        test_mean_accuracy = multiclass_mean_accuracy(output_test, test_labels)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {loss_train.item():.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {test_accuracy:.4f}, "
          f"Test Mean Accuracy: {test_mean_accuracy:.4f}", flush=True)

    # Inside the epoch loop, after calculating test_mean_accuracy
    if test_mean_accuracy > best_mean_test_accuracy:
        best_mean_test_accuracy = test_mean_accuracy
        torch.save(model.state_dict(), f"best_model_bce.pth")
    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"model_bce_{epoch}.pth")

# save model
print('saved', best_test_accuracy)
