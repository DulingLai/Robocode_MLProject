package neuralNet;

import java.io.*;
import java.util.HashMap;
import java.util.Random;

import interfaces.NeuralNetInterface;

import static java.lang.Double.isNaN;

public class NeuralNet implements NeuralNetInterface {

    // param inherited from the interface
    private int argNumInputs;
    private int argNumHidden;
    private int argNumOutputs;
    private double argLearningRate;
    private double argMomentumTerm;
    private double argA;
    private double argB;
    private boolean argUseBipolarHiddenNeurons;

    /*
    Hidden neurons, inputs, output, expectedOutput
    */
    private double[] hiddenNeuron;
    private double[] inputNeuron;
    private double[] outputNeuron;
    private double[] expectedOutput;

    /*
    array to store the weight of connections
     */
    private double[][] weightInputHidden;   // +1 for bias neuron
    private double[][] weightHiddenOutput;

    // store the previousDeltaWeight for each connection
    private double[][] deltaWeightInputHidden;
    private double[][] deltaWeightHiddenOutput;
    private double[][] prevInputHiddenDeltaWeight;
    private double[][] prevHiddenOutputDeltaWeight;

    // errors - unit neuron
    private double[] erro;
    private double[] errh;


    /**
     * Constructor. (Cannot be declared in an interface, but your implementation will need one)
     *
     * @param argNumInputs               The number of inputs in your input vector
     * @param argNumHidden               The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argNumOutputs              The number of outputs in the output vector
     * @param argLearningRate            The learning rate coefficient
     * @param argMomentumTerm            The momentum coefficient
     * @param argA                       Integer lower bound of sigmoid used by the output neuron only.
     * @param argB                       Integer upper bound of sigmoid used by the output neuron only.
     * @param argUseBipolarHiddenNeurons boolean to use bipolar hidden neurons.
     */
    public NeuralNet(
            int argNumInputs,
            int argNumHidden,
            int argNumOutputs,
            double argLearningRate,
            double argMomentumTerm,
            double argA,
            double argB,
            boolean argUseBipolarHiddenNeurons) {
        // store parameters
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutputs = argNumOutputs;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.argA = argA;
        this.argB = argB;
        this.argUseBipolarHiddenNeurons = argUseBipolarHiddenNeurons;

        hiddenNeuron = new double[argNumHidden];
        inputNeuron = new double[argNumInputs];
        outputNeuron = new double[argNumOutputs];
        expectedOutput = new double[argNumOutputs];

        weightInputHidden = new double[argNumInputs+1][argNumHidden];
        weightHiddenOutput = new double[argNumHidden+1][argNumOutputs];

        deltaWeightInputHidden = new double[argNumInputs+1][argNumHidden];
        deltaWeightHiddenOutput = new double[argNumHidden+1][argNumOutputs];
        prevInputHiddenDeltaWeight = new double[argNumInputs+1][argNumHidden];
        prevHiddenOutputDeltaWeight = new double[argNumHidden+1][argNumOutputs];

        erro = new double[argNumOutputs];
        errh = new double[argNumHidden];
    }

    /**
     * get a random number
     * @return random number
     */
    private double getRandom(){
        final Random rand = new Random();
        return (rand.nextDouble() - 0.5) * 0.01; // [-0.005;0.005]
    }

    /**
     * sigmoid function
     * @param x The input
     * @return The sigmoid output of x
     */
    @Override
    public double sigmoid(double x) {
        if(argUseBipolarHiddenNeurons){
            return 2/(1 + Math.exp(-x))-1;
        } else {
            return 1/(1 + Math.exp(-x));
        }
    }

    @Override
    public double customSigmoid(double x) {
        return 2*argB/(1 + Math.exp(-x))+argA;
    }

    private double sigmoidDerivative(final double val){
        if(Constants.CUSTOM_SIGMOID) return ((1.0 + val) * (1.0 - val));
        else if(argUseBipolarHiddenNeurons) return (0.5 * (1.0 + val) * (1.0 - val));
        else return (val * (1.0 - val));
    }

    @Override
    public void initializeWeights() {
        // initialize weight for connections between hidden layer and input layer
        for (int input = 0; input < argNumInputs + 1; input++){         // +1 for bias
            for(int hidden = 0; hidden < argNumHidden; hidden++) {
                weightInputHidden[input][hidden] = getRandom();
                // initialize the previousDeltaWeight to 0
                prevInputHiddenDeltaWeight[input][hidden] = 0.0;
            }
        }

        // initialize weight for connections between output layer and hidden layer
        for (int hidden = 0; hidden < argNumHidden + 1; hidden++) {     // +1 for bias
            for(int output = 0; output < argNumOutputs; output++) {
                weightHiddenOutput[hidden][output] = getRandom();
                // initialize the previousDeltaWeight to 0
                prevHiddenOutputDeltaWeight[hidden][output] = 0.0;
            }
        }
    }

    /**
     * set all connections' weight to the same value (i.e. 0 to clear all weight)
     * @param weight The weight of all connections to be set
     */
    @Override
    public void setAllWeights(double weight) {
        // initialize weight for connections between hidden layer and input layer
        for (int input = 0; input < argNumInputs + 1; input++){         // +1 for bias
            for(int hidden = 0; hidden < argNumHidden; hidden++) {
                weightInputHidden[input][hidden] = weight;
                // initialize the previousDeltaWeight to 0
                prevInputHiddenDeltaWeight[input][hidden] = 0.0;
            }
        }

        // initialize weight for connections between output layer and hidden layer
        for (int hidden = 0; hidden < argNumHidden + 1; hidden++) {     // +1 for bias
            for(int output = 0; output < argNumOutputs; output++) {
                weightHiddenOutput[hidden][output] = weight;
                // initialize the previousDeltaWeight to 0
                prevHiddenOutputDeltaWeight[hidden][output] = 0.0;
            }
        }
    }

    /**
     * get the output for a given input using the neural network.
     * @param X The input vector. An array of doubles.
     * @return the output of the neural network.
     */
    @Override
    public double outputFor(double[] X) {
        feedforward(X);
        return outputNeuron[0];
    }

    /**
     * train the neural network with the input and expected input
     * @param X        The input vector
     * @param argValue The new value to learn
     * @return The output of the neural network
     */
    @Override
    public double train(double[] X, double argValue) {

        // feed forward all the values
        feedforward(X);

        // perform error back propagation - argValue is the expected result
        backPropagation(argValue);

        return outputNeuron[0];
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    /**
     * apply the feed forward of the input
     * @param X The input arrays.
     */
    private void feedforward(double[] X){
        double sum;

        // store inputs into input neurons
        for(int i = 0; i<argNumInputs; i++) {
            inputNeuron[i] = X[i];
        }

        // calculate input to the hidden neurons, and store the output of the hidden neurons
        for(int hidden = 0; hidden < argNumHidden; hidden++){   // loop over all neurons in the hidden layer
            sum = 0.0;
            for(int input = 0; input < argNumInputs; input++) { // loop over all neurons in the input layer
                sum += inputNeuron[input] * weightInputHidden[input][hidden];
            }
            sum += weightInputHidden[argNumInputs][hidden];     // add bias
            if(Constants.CUSTOM_SIGMOID) hiddenNeuron[hidden] = customSigmoid(sum);
            else hiddenNeuron[hidden] = sigmoid(sum);
        }

        // calculate input to the output neurons, and store the output of the output neurons
        for(int out = 0; out < argNumOutputs; out++){   // loop over all neurons in the output layer
            sum = 0.0;
            for(int hid = 0; hid < argNumHidden; hid++) { // loop over all neurons in the hidden layer
                sum += hiddenNeuron[hid] * weightHiddenOutput[hid][out];
            }
            sum += weightHiddenOutput[argNumHidden][out];     // add bias
            if(Constants.CUSTOM_SIGMOID) outputNeuron[out] = customSigmoid(sum);
            else outputNeuron[out] = sigmoid(sum);
        }
    }

    /**
     * apply error back propagation of the neural network.
     * @param expectedOutputs The expected output for the neural network.
     */
    private void backPropagation(double expectedOutputs){
        // calculate the output layer error
        for(int out = 0; out < argNumOutputs; out++) {
            // compute the error information term for each output neuron
            erro[out] = (expectedOutputs - outputNeuron[out]) * sigmoidDerivative(outputNeuron[out]);      // step 6.1 error information
            // compute the weight correction term
            for(int hid = 0; hid < argNumHidden; hid++) {
                deltaWeightHiddenOutput[hid][out] = argLearningRate * erro[out] * hiddenNeuron[hid];        // step 6.2 weight correction
            }
            deltaWeightHiddenOutput[argNumHidden][out] = argLearningRate * erro[out];       // step 6.3 bias weight correction term
        }

        // calculate the hidden layer error
        for(int hid = 0; hid < argNumHidden; hid++) {       // loop through all hidden neurons
            double sumDeltaInput = 0.0;
            for(int out = 0; out < argNumOutputs; out++) {
                sumDeltaInput += erro[out] * weightHiddenOutput[hid][out];      // step 7.1 delta inputs
            }
            errh[hid] = sumDeltaInput * sigmoidDerivative(hiddenNeuron[hid]);      // step 7.2  hidden neuron error information term
            // compute weight correction term
            for(int in = 0; in < argNumInputs; in++){
                deltaWeightInputHidden[in][hid] = argLearningRate * errh[hid] * inputNeuron[in];    // step 7.3 weight correction
            }
            deltaWeightInputHidden[argNumInputs][hid] = argLearningRate * errh[hid];       // step 7.4 bias weight correction
        }

        // Update the weights for hidden/output connection.
        for(int out = 0; out < argNumOutputs; out++) {
            for(int hid = 0; hid < argNumHidden; hid++) {
                weightHiddenOutput[hid][out] += deltaWeightHiddenOutput[hid][out] + argMomentumTerm * prevHiddenOutputDeltaWeight[hid][out];
                prevHiddenOutputDeltaWeight[hid][out] = deltaWeightHiddenOutput[hid][out];
            }
            weightHiddenOutput[argNumHidden][out] += deltaWeightHiddenOutput[argNumHidden][out] + argMomentumTerm * prevHiddenOutputDeltaWeight[argNumHidden][out]; // Update the bias.
            prevHiddenOutputDeltaWeight[argNumHidden][out] = deltaWeightHiddenOutput[argNumHidden][out];
        }

        // Update the weights for input/hidden connections
        for(int hid = 0; hid < argNumHidden; hid++) {
            for(int in = 0; in < argNumInputs; in++) {
                weightInputHidden[in][hid] += deltaWeightInputHidden[in][hid] + argMomentumTerm * prevInputHiddenDeltaWeight[in][hid];
                prevInputHiddenDeltaWeight[in][hid] = deltaWeightInputHidden[in][hid] * 1;
            }
            weightInputHidden[argNumInputs][hid] += deltaWeightInputHidden[argNumInputs][hid] + argMomentumTerm * prevInputHiddenDeltaWeight[argNumInputs][hid]; // Update the bias.
            prevInputHiddenDeltaWeight[argNumInputs][hid] = deltaWeightInputHidden[argNumInputs][hid] * 1;
        }
    }

    /**
     * print the training result to the console or a txt file
     * @param inputs
     * @param expectedOutputs
     * @param actualResult
     * @param error
     * @param epoch
     * @param saveToFile
     */
    public void printTrainResult(double[][] inputs, double expectedOutputs[], double actualResult[], double error, int epoch, boolean saveToFile, String fileName) {

        // print the training result to the console
        System.out.print("Epoch: " + epoch +", ");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("INPUTS: ");
            for (int x = 0; x < argNumInputs; x++) {
                System.out.print(inputs[p][x] + ", ");
            }
            System.out.print("EXPECTED: ");
            System.out.print(expectedOutputs[p] + ", ");

            System.out.print("ACTUAL: ");
            System.out.print(actualResult[p] + ", \n");

        }
        System.out.print("ERROR: ");
        System.out.print(error + "; ");
        System.out.println();

        // print the training result to a txt file
        if(saveToFile) {
            try {
                FileWriter fw = new FileWriter(Constants.RESULT_FILE_PATH+fileName, true);
                fw.write(epoch + ", " + error + "\n");
                fw.flush();
                fw.close();
            } catch(IOException e){
                System.out.println("Error reading text file! Please check the file path in code!");
            }
        }


    }


}


