import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from compute_accuracy import compute_accuracy

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]        
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

if __name__ == "__main__":
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])
    
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    train_len = len(train_ds)
    print("train_len ", train_len)

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
        )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
        )
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params) #2*30+30*20+20*2+30+20+2=752
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)
            loss = F.cross_entropy(logits, labels) # Loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {loss:.2f}")
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
        print(outputs)
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)
    predictions = torch.argmax(probas, dim=1)
    print(predictions)
    prediction_compare = predictions == y_train
    print(prediction_compare)
    computed_train_accuracy = compute_accuracy(model, train_loader)
    print("computed_train_accuracy ", computed_train_accuracy)
    computed_test_accuracy = compute_accuracy(model, test_loader)
    print("computed_test_accuracy ", computed_test_accuracy)

    