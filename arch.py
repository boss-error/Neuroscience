from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

class Neuron():
    def __init__(self, x, y, value=0.0):
        self.x = x
        self.y = y
        self.value = value  # Store neuron value

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        # Display neuron value
        pyplot.text(self.x, self.y, f"{self.value:.2f}", fontsize=12, ha='center', va='center')

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, values=None):
        self.vertical_distance_between_layers = 4  # Increased for clarity
        self.horizontal_distance_between_neurons = 6  # Increased for spacing
        self.neuron_radius = 0.7
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, values)

    def __intialise_neurons(self, number_of_neurons, values):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for i in range(number_of_neurons):
            value = values[i] if values is not None else 0.0  # Get value if provided
            neuron = Neuron(x, self.y, value)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight, index=0, total=1):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), 
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

        # Adjust weight position to prevent overlap
        offset = (index - (total - 1) / 2) * .9
        mid_x = (neuron1.x + neuron2.x) / 2
        mid_y = (neuron1.y + neuron2.y) / 2 + offset
        pyplot.text(mid_x, mid_y, f"{weight:.2f}", fontsize=10, ha='center', va='center', color='blue')

    def draw(self, layerType=0, weights=None):
        for i, neuron in enumerate(self.neurons):
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                total_connections = len(self.previous_layer.neurons)
                for j, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    weight = weights[i][j] if weights is not None else np.random.uniform(-0.5, 0.5)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight, j, total_connections)

        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        layer_name = "Input Layer" if layerType == 0 else "Output Layer" if layerType == -1 else f"Hidden Layer {layerType}"
        pyplot.text(x_text, self.y, layer_name, fontsize=12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []

    def add_layer(self, number_of_neurons, values=None, weights=None):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, values)
        self.layers.append(layer)

    def draw(self, weights=None):
        pyplot.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.draw(i if i < len(self.layers)-1 else -1, weights[i-1] if weights and i > 0 else None)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network Architecture', fontsize=15)
        pyplot.show()

class DrawNN():
    def __init__(self, neural_network, values, weights):
        self.neural_network = neural_network
        self.values = values
        self.weights = weights

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for i, l in enumerate(self.neural_network):
            network.add_layer(l, self.values[i], self.weights[i-1] if i > 0 else None)
        network.draw(self.weights)

# Define network structure
network_structure = [2, 2, 2]  # Input, hidden, and output layers



values = [[i1, i2], [out1, out2], [out3, out4]]
weights = [[[w1, w2], [w3, w4]], [[w5, w6], [w7, w8]]]

# Draw network
network = DrawNN(network_structure, values, weights)
network.draw()
