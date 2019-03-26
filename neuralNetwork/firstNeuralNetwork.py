import numpy as np

class NeuralNetwork:
    def __init__(self, input_data, actual_output):
        self.input         = input_data
        self.weights1      = np.random.rand(input_data.shape[1], input_data.shape[0]) 
        self.weights2      = np.random.rand(input_data.shape[0], 1)                 
        self.actual_output = actual_output
        self.output        = np.zeros(actual_output.shape)

    def feedforward(self):
        self.layer1 = NeuralNetwork.sigmoid(np.dot(self.input, self.weights1))
        self.output = NeuralNetwork.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.actual_output - self.output) * NeuralNetwork.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.actual_output - self.output) * NeuralNetwork.sigmoid_derivative(self.output), self.weights2.T) * NeuralNetwork.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    @staticmethod    
    def sigmoid(x):
        return 1.0/(1+ np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1.0 - x)

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(60000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
