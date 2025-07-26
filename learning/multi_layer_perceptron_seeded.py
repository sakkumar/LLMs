import torch

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


if __name__ == "__main__":
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params) #50*30+30*20+20*3+30+20+3=2213
    print(model.layers[0].weight)
    print(model.layers[0].weight.shape)
    print(model.layers[0].bias)
    print(model.layers[0].bias.shape)
    print(model.layers[2].weight)
    print(model.layers[2].weight.shape)
    print(model.layers[2].bias)
    print(model.layers[2].bias.shape)
    print(model.layers[4].weight)
    print(model.layers[4].weight.shape)
    print(model.layers[4].bias)
    print(model.layers[4].bias.shape)
    torch.manual_seed(123)
    x = torch.rand((1, 50))
    out = model(x)
    print("training: ", out)
    with torch.no_grad():
        out = model(x)
    print("inference: ", out)
    with torch.no_grad():
        out = torch.softmax(model(x), dim=1)
    print("softmax inference: ", out)
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    print("sigmoid inference: ", out)




