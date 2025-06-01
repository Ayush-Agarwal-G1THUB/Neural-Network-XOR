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
    vector<double> weightsIn;
    vector<double> wGradients;
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
 * @param nodesPerLayer Input : Stores the number of nodes in each layer of the NN
 * @param num_layers Stores the number of layers in the NN (including input, hidden and output)
 */
void createNN (vector<layer> &nn, const vector<int> &nodesPerLayer, int &num_layers) {
    num_layers = nodesPerLayer.size();
    nn.resize(num_layers);

    for (int layerIdx = 0; layerIdx<num_layers; layerIdx++) {
        nn[layerIdx].resize(nodesPerLayer[layerIdx]);

        if (layerIdx != 0) {    
            layer& layer = nn[layerIdx];

            for (node &node : layer) {
                node.weightsIn.resize(nn[layerIdx-1].size());
                node.wGradients.resize(nn[layerIdx-1].size());
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
    for (int layerIdx = 1; layerIdx<nn.size(); layerIdx++) {
        layer& layer = nn[layerIdx];
        for (node& node : layer) {
            for (double& w : node.weightsIn) {
                w = initRand();
            }
            node.bias = initRand();
        }
    }
}

/**
 * @brief Returns the z = sum(xi.wij)+bj for the node j
 * @param nn Neural Network
 * @param layerIdx Layer index in the neural network
 * @param nodeIdx Node index in the layer
 */
double calculateZ (vector<layer> &nn, int layerIdx, int nodeIdx) {
    // Error handling
    if (layerIdx == 0) {
        cout << "Error : Trying to compute z value for input layer" << endl;
        return -1;
    }

    node& currNode = nn[layerIdx][nodeIdx];
    layer& prevLayer = nn[layerIdx-1];

    double z = 0;

    for (int i = 0; i<prevLayer.size(); i++) {
        node& prevNode = prevLayer[i];
        z += prevNode.activation * currNode.weightsIn[i];
    }

    z += currNode.bias;

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

    for (int layerIdx = 1; layerIdx<num_layers; layerIdx++) {
        layer& layer = nn[layerIdx];
        for (int nodeIdx = 0; nodeIdx < layer.size(); nodeIdx++) {
            node& node = layer[nodeIdx];

            node.z = calculateZ(nn, layerIdx, nodeIdx);
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
    for (int layeridx = num_layers-1; layeridx>0; layeridx--) {
        layer &currLayer = nn[layeridx];
        layer &prevLayer = nn[layeridx-1];

        // Compute current layer error signals
        if (layeridx == num_layers-1) { // Output layer
            // Calculated directly using the formula
            layer& outputLayer = nn[num_layers-1];
            node& outputNode = outputLayer[0];
            outputNode.error_signal = outputNode.activation - y;
        }
        // For hidden layers, summation(deltaK.wHK) has already been computed by the forward layer
        // For output layer, (activation-label) has already been manually computed
        for (node& node : currLayer) {
            node.error_signal *= node.activation * (1 - node.activation);
        }

        for (node& node : prevLayer) node.error_signal = 0; 
        // Start the error signal in all nodes of the previous layer to 0
        // So that error_signal * weight can be summed as below

        // In the nodes of currLayer, the error signal currently has the value of summation(error_signalK * weightHK)
        // Precomputed when parsing the forward layer
        // Or it is the output layer and has the manually computed error_signal
        for (node& currNode : currLayer) {
            // Update weights of current node
            for (int i = 0; i<prevLayer.size(); i++) {
                double& wGradientIJ = currNode.wGradients[i];
                wGradientIJ = currNode.error_signal * prevLayer[i].activation; // delJ/delW

                currNode.weightsIn[i] -= learning_rate * wGradientIJ; // Update weight for current node

                prevLayer[i].error_signal += currNode.error_signal * currNode.weightsIn[i]; 
                // Summation step to update error_signal for the previous layer 
            }

            // Update bias of current node
            double biasGradient = currNode.error_signal;
            currNode.bias -= learning_rate * biasGradient;
        }
    }
}

/**
 * @brief Driver code for training the neural network
 * @param num_epochs Input : The number of times the neural network goes through the entire training_set
 * @param nn Output : The trained neural network
 */
void trainNN (const int num_layers, vector<point> training_set, vector<layer> &nn, const int num_epochs, const double learning_rate) {
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
    int correct = 0;

    for (point point : test_set) {
        int a = point.feature1;
        int b = point.feature2;

        forwardPropagate(num_layers, nn, a, b);

        double outputVal = nn[num_layers-1][0].activation;
        int prediction = (outputVal > 0.5) ? 1 : 0;

        int label = point.label;

        if (prediction == label) {
            // cout << "Correct" << endl;
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
    const int test_size = 100;
    const int num_epochs = 10000;
    const double learning_rate = 0.01;
    
    const vector<int> nodesPerLayer = {2, 2, 1};
    vector<point> training_set;
    vector<point> test_set;
    vector<layer> nn;

    int num_layers; // Including input, hidden and output layers

    loadSets(dataset, training_set, training_size, test_set, test_size);
    createNN(nn, nodesPerLayer, num_layers);
    initNN(nn);
    trainNN(num_layers, training_set, nn, num_epochs, learning_rate);
    evaluateNN(nn, test_set, test_size, num_layers);

    return 0;
}