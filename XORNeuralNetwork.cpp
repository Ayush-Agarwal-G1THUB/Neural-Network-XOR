#include <iostream>
#include <string>
#include <vector>
#include <random>

using namespace std;

struct point {
    int feature1;
    int feature2;
    int label;
};

struct node {
    double z;
    double activation;
    double bias;
    double error_signal;
    vector<double> weights_in;
    vector<double> w_gradients;
};

using layer = vector<node>;

/**
 * @brief Loads the dataset into training and test sets.
 * @param dataset Input : Dataset used to load into training and test sets.
 * @param training_size Input : Size of training_set.
 * @param test_size Input : Size of test_size.
 * @param training_set Output : Vector to store training data.
 * @param test_set Output : Vector to store test data.
 */
void loadSets (vector<point>& dataset, vector<point>& training_set, const int training_size, vector<point>& test_set, const int test_size) {
    training_set.resize(training_size);
    test_set.resize(test_size);

    // Load random data points from the dataset into the training set
    for (int i = 0; i<training_size; i++) {
        int idx = rand() % 4;
        training_set[i] = dataset[idx];
    }

    // Load random data points from the dataset into the test set
    for (int i = 0; i<test_size; i++) {
        int idx = rand() % 4;
        test_set[i] = dataset[idx];
    }
}

/**
 * @brief Creates the NN
 * @param nn Output : Stores the vector of layers, each a vector of nodes
 * @param nodes_per_layer Input : Stores the number of nodes in each layer of the NN
 * @param num_layers Stores the number of layers in the NN (including input, hidden and output)
 */
void createNN (vector<layer> &nn, const vector<int> &nodes_per_layer, int &num_layers) {
    num_layers = nodes_per_layer.size();
    nn.resize(num_layers);

    for (int layer_idx = 0; layer_idx<num_layers; layer_idx++) {
        nn[layer_idx].resize(nodes_per_layer[layer_idx]);

        if (layer_idx != 0) {    
            layer& layer = nn[layer_idx];

            for (node &node : layer) {
                node.weights_in.resize(nn[layer_idx-1].size());
                node.w_gradients.resize(nn[layer_idx-1].size());
            }
        }
        
    }
}

/**
 * @brief Returns a random double between -1 and +1
 */
double initRand () {
    return (((double)rand()/RAND_MAX)-0.5)*2;
}

/**
 * @brief Populates the weights and biases of the neural network with random values between -1 and 1
 * @param nn Stores the neural network
 */
void initNN (vector<layer> &nn) {
    cout << "Initialising neural network..." << endl;

    for (int layer_idx = 1; layer_idx<nn.size(); layer_idx++) {
        layer& layer = nn[layer_idx];
        for (node& node : layer) {
            for (double& w : node.weights_in) {
                w = initRand();
            }
            node.bias = initRand();
        }
    }
}

/**
 * @brief Returns the z = sum(xi.wij)+bj for the node j
 * @param nn Neural Network
 * @param layer_idx Layer index in the neural network
 * @param node_idx Node index in the layer
 */
double calculateZ (vector<layer> &nn, int layer_idx, int node_idx) {
    // Error handling
    if (layer_idx == 0) {
        cout << "Error : Trying to compute z value for input layer" << endl;
        return -1;
    }

    node& curr_node = nn[layer_idx][node_idx];
    layer& prev_layer = nn[layer_idx-1];

    double z = 0;

    for (int i = 0; i<prev_layer.size(); i++) {
        node& prev_node = prev_layer[i];
        z += prev_node.activation * curr_node.weights_in[i];
    }

    z += curr_node.bias;

    return z;
}

/**
 * @brief Returns f(x) = sigmoid of x.
 * @param x Input : Input value x.
 */
double sigmoid (double x) {
    return 1 / (1 + exp(-x));
}

/**
 * @brief Conducts forward propagation for a given input.
 * @param a Input : Feature 1.
 * @param b Input : Feature 2.
 */
void forwardPropagate (const int num_layers, vector<layer> &nn, double a, double b) {
    nn[0][0].activation = a;
    nn[0][1].activation = b;

    for (int layer_idx = 1; layer_idx<num_layers; layer_idx++) {
        layer& layer = nn[layer_idx];
        for (int node_idx = 0; node_idx < layer.size(); node_idx++) {
            node& node = layer[node_idx];

            node.z = calculateZ(nn, layer_idx, node_idx);
            node.activation = sigmoid(node.z);
        }
    }
}

/**
 * @brief Conductions backpropagation after forward propagation has been carried out
 * @param y Input : The correct label for the input of the current training example
 * @param learning_rate Input : Decides how fast the neural network learns
 */
void backPropagate (vector<layer> &nn, const int num_layers, const int y, const double &learning_rate) {
    for (int layer_idx = num_layers-1; layer_idx>0; layer_idx--) {
        layer &curr_layer = nn[layer_idx];
        layer &prev_layer = nn[layer_idx-1];

        // Compute current layer error signals
        if (layer_idx == num_layers-1) { // Output layer
            // Calculated directly using the formula
            layer& output_layer = nn[num_layers-1];
            node& output_node = output_layer[0];
            output_node.error_signal = output_node.activation - y;
        }
        // For hidden layers, summation(deltaK.wHK) has already been computed by the forward layer
        // For output layer, (activation-label) has already been manually computed
        for (node& node : curr_layer) {
            node.error_signal *= node.activation * (1 - node.activation);
        }

        for (node& node : prev_layer) node.error_signal = 0; 
        // Start the error signal in all nodes of the previous layer to 0
        // So that error_signal * weight can be summed as below

        // In the nodes of curr_layer, the error signal currently has the value of summation(error_signalK * weightHK)
        // Precomputed when parsing the forward layer
        // Or it is the output layer and has the manually computed error_signal
        for (node& curr_node : curr_layer) {
            // Update weights of current node
            for (int i = 0; i<prev_layer.size(); i++) {
                double& w_gradient_ij = curr_node.w_gradients[i];
                w_gradient_ij = curr_node.error_signal * prev_layer[i].activation; // delJ/delW

                curr_node.weights_in[i] -= learning_rate * w_gradient_ij; // Update weight for current node

                prev_layer[i].error_signal += curr_node.error_signal * curr_node.weights_in[i]; 
                // Summation step to update error_signal for the previous layer 
            }

            // Update bias of current node
            double bias_gradient = curr_node.error_signal;
            curr_node.bias -= learning_rate * bias_gradient;
        }
    }
}

/**
 * @brief Driver code for training the neural network
 * @param num_epochs Input : The number of times the neural network goes through the entire training_set
 * @param nn Output : The trained neural network
 */
void trainNN (const int num_layers, vector<point> training_set, vector<layer> &nn, const int num_epochs, const double learning_rate) {
    cout << "Training..." << endl;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (point point : training_set) {
            int a = point.feature1;
            int b = point.feature2;
            int y = point.label;

            forwardPropagate(num_layers, nn, a, b);
            backPropagate(nn, num_layers, y, learning_rate);
        }
    }
}

/**
 * @brief Evaluates the performance of the neural netword
 */
void evaluateNN (vector<layer>& nn, vector<point>& test_set, const int& test_size, const int num_layers) {
    cout << "Evaluating..." << endl;
    int correct = 0;

    for (point point : test_set) {
        int a = point.feature1;
        int b = point.feature2;

        forwardPropagate(num_layers, nn, a, b);

        double output_val = nn[num_layers-1][0].activation;
        int prediction = (output_val > 0.5) ? 1 : 0;

        int label = point.label;

        if (prediction == label) {
            /* cout << "Correct" << endl;
            cout << "Input Layer Node 1 = " << a << endl;
            cout << "Input Layer Node 2 = " << b << endl;
            cout << "Hidden Layer Node 1 = " << nn[1][0].activation << endl;
            cout << "Hidden Layer Node 2 = " << nn[1][1].activation << endl;
            cout << "Output Layer Node 1 = " << output_val << "\n" << endl; */
            correct++;
        }
        else {
            cout << "Wrong" << endl;
            cout << "Inputs     = " << a << " " << b << endl;
            cout << "Prediction = " << prediction << endl;
            cout << "Actual ans = " << label << "\n" << endl;
        }
    }

    cout << "Test size = " << test_size << endl;
    cout << "Correct   = " << correct << endl;
    cout << "Accuracy  = " << 100 * (double)correct/test_size << "%" << endl;
}

int main()
{
    srand(time(0));

    vector<point> dataset = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };
    
    const int training_size = 1000;
    const int test_size = 50;
    const int num_epochs = 10000;
    const double learning_rate = 0.01;
    
    const vector<int> nodes_per_layer = {2, 2, 1};
    vector<point> training_set;
    vector<point> test_set;
    vector<layer> nn;

    int num_layers; // Including input, hidden and output layers

    loadSets(dataset, training_set, training_size, test_set, test_size);
    createNN(nn, nodes_per_layer, num_layers);
    initNN(nn);
    trainNN(num_layers, training_set, nn, num_epochs, learning_rate);
    evaluateNN(nn, test_set, test_size, num_layers);

    return 0;
}
