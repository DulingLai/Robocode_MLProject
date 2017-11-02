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
    HashMap to store the net input and output of the neurons
    */
    private HashMap<Integer, Double> neuronOutput = new HashMap<>();

    /*
    HashMap to store the output of input neurons
    Integer 01, 02
     */
    private HashMap<Integer, Double> inputs = new HashMap<>();

    /*
     HashMap to store the weight of connections
     Four digits number 1234
     1, 3 - indicates the layer: 0 for input, 1 for hidden, 2 for output
     2, 4 - indicates the neuron, range: 0 for bias, 1...2 for input, 1...4 for hidden, 1 for output
     */
    private HashMap<Integer, Double> connectionWeight = new HashMap<>();

    // HashMap to store the deltaWeight and previousDeltaWeight for each connection
    private HashMap<Integer, Double> deltaWeight = new HashMap<>();
    private HashMap<Integer, Double> previousDeltaWeight = new HashMap<>();


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
    }

    /**
     * get a random number
     * @return random number
     */
    private double getRandom(){
        final Random rand = new Random();
        return rand.nextDouble() - 0.5; // [-0.5;0.5]
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
        // not used for this part
        return 0;
    }

    @Override
    public void initializeWeights() {
        // initialize weight for connections between hidden layer and input layer
        for(int p = 11; p < 11+argNumHidden; p++) {
            for (int i = 0; i < argNumInputs+1; i++){
                connectionWeight.put(p*100 + i, getRandom());

                // initialize the previousDeltaWeight to 0
                previousDeltaWeight.put(p*100 + i, 0.0);
            }
        }

        // initialize weight for connections between output layer and hidden layer
        for(int p = 21; p < 21+argNumOutputs; p++) {
            for (int i = 10; i < 11 + argNumHidden; i++) {
                connectionWeight.put(p*100 + i, getRandom());

                // initialize the previousDeltaWeight to 0
                previousDeltaWeight.put(p*100 + i, 0.0);
            }
        }

    }

    /**
     * set all connections' weight to the same value (i.e. 0 to clear all weight)
     * @param weight The weight of all connections to be set
     */
    @Override
    public void setAllWeights(double weight) {
        // Zero weight for connections between hidden layer and input layer
        for(int p = 11; p <= 10+argNumHidden; p++) {
            for (int i = 0; i <= argNumInputs+1; i++) {
                connectionWeight.put(p*100 + i, weight);
            }
        }

        // Zero weight for connections between output layer and hidden layer
        for(int p = 21; p <= 20+argNumOutputs; p++) {
            for (int i = 10; i <= 10 + argNumHidden; i++) {
                connectionWeight.put(p * 100 + i, weight);
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
        return neuronOutput.get(21);
    }

    /**
     * train the neural network with the input and expected input
     * @param X        The input vector
     * @param argValue The new value to learn
     * @return The output of the neural network
     */
    @Override
    public double train(double[] X, double argValue) {

        feedforward(X);

        backPropagation(argValue);

        if(!Constants.IMMEDIATE_WEIGHT_UPDATE) {
            updateWeight();
        }

        return neuronOutput.get(21);
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    /**
     * apply the feed forward of the input
     * @param input The input arrays.
     */
    private void feedforward(double[] input){
        // store inputs into the HashMap
        for(int i = 0; i<argNumInputs; i++) {
            inputs.put(i+1, input[i]);
        }

        // calculate and store the output into HashMap
        for(int i=11; i < 11+argNumHidden; i++){ // loop over all neurons in the hidden layer ID = 11, 12, 13, 14
            calculateOutput(i);
        }
        for(int i=21; i < 21+argNumOutputs; i++){ // loop over all output neurons ID = 21
            calculateOutput(i);
        }
    }

    /**
     * calculate the output for a given neuron
     * @param i The id index of the neuron, 1x for hidden layer, 2x for output layer
     */
    private void calculateOutput(int i) {
        double z_in = 0.0;
        if(10<i && i<11+argNumHidden) { // hidden layer
            for(int j = 1; j <= argNumInputs; j++) {
                z_in += connectionWeight.get(i * 100 + j) * inputs.get(j);
            }
            z_in += connectionWeight.get(1100);

            // activation function
            z_in = sigmoid(z_in);

            // store the output in the HashMap
            neuronOutput.put(i, z_in);

        } else if (20<i && i<21+argNumOutputs){ // output layer
            for(int j = 11; j < 11 + argNumHidden; j++) {
                z_in += connectionWeight.get(i * 100 + j) * neuronOutput.get(j);
            }
            z_in += connectionWeight.get(2110);

            // activation function
            z_in = sigmoid(z_in);
            neuronOutput.put(i, z_in);
        } else {
            throw new java.lang.RuntimeException("This neuron id is out of range!");
        }
    }

    /**
     * apply error back propagation of the neural network.
     * @param expectedOutput The expected output for the neural network.
     */
    private void backPropagation(Double expectedOutput){

        int output_neuron_id = 21;
        double y = neuronOutput.get(output_neuron_id);
        double output_error_infoTerm;

        // compute the error information term for each output neuron
        if(argUseBipolarHiddenNeurons){
            output_error_infoTerm = (expectedOutput - y) * 0.5 * (1 - Math.pow(y,2));
        } else {
            output_error_infoTerm = (expectedOutput - y) * y * (1 - y);
        }

        // compute the weight correction term for output/bias connection
        deltaWeight.put(2110, argLearningRate * output_error_infoTerm);

        // immediate weight update
        if(Constants.IMMEDIATE_WEIGHT_UPDATE){
            connectionWeight.put(2110, connectionWeight.get(2110) + deltaWeight.get(2110) + argMomentumTerm * previousDeltaWeight.get(2110));
            previousDeltaWeight.put(2110, deltaWeight.get(2110));
        }

        for(int i=11; i<11+argNumHidden; i++) { // loop through the hidden layers
            // compute the weight correction term for each output/hidden connection
            deltaWeight.put(2100+i, argLearningRate * output_error_infoTerm * neuronOutput.get(i));

            if(Constants.IMMEDIATE_WEIGHT_UPDATE){
                connectionWeight.put(2100+i, connectionWeight.get(2100+i)+deltaWeight.get(2100+i)+ argMomentumTerm * previousDeltaWeight.get(2100+i));
                previousDeltaWeight.put(2100, deltaWeight.get(2100+i));;
            }

            // delta input for hidden neurons from the output layer
            double deltaInput = output_error_infoTerm * connectionWeight.get(2100+i);
            double zj = neuronOutput.get(i);
            double hidden_error_infoTerm;
            if(argUseBipolarHiddenNeurons) {
                hidden_error_infoTerm = deltaInput * 0.5 * (1 - Math.pow(zj,2));
            } else {
                hidden_error_infoTerm = deltaInput *  zj * (1 - zj);
            }

            // calculate the weight correction for hidden/input connection
            for(int j=1; j<1+argNumInputs; j++) {
                deltaWeight.put(i * 100 + j, argLearningRate * hidden_error_infoTerm * inputs.get(j));
                if(Constants.IMMEDIATE_WEIGHT_UPDATE){
                    connectionWeight.put(i * 100 + j, connectionWeight.get(i * 100 + j) + deltaWeight.get(i * 100 + j)+ argMomentumTerm * previousDeltaWeight.get(i * 100 + j));
                    previousDeltaWeight.put(i * 100 + j, deltaWeight.get(i * 100 + j));
                }
            }

            // compute the delta weight for the hidden/bias connections
            deltaWeight.put(i*100, argLearningRate * hidden_error_infoTerm);

            if(Constants.IMMEDIATE_WEIGHT_UPDATE){
                connectionWeight.put(i*100, connectionWeight.get(i*100) + deltaWeight.get(i*100) + argMomentumTerm * previousDeltaWeight.get(i * 100));
                previousDeltaWeight.put(i * 100, deltaWeight.get(i * 100));
            }
        }
    }


    /**
     * Update the weight of all connections, and store the previous delta weight
     */
    private void updateWeight(){
        // update all connection weight
        connectionWeight.replaceAll((key, value) -> value + deltaWeight.get(key) + argMomentumTerm * previousDeltaWeight.get(key));

        // update the previous delta weight
        previousDeltaWeight.replaceAll((key,value) -> deltaWeight.get(key));
    }

    /**
     *  getter for weight (for testing)
     * @param id The id of the connection
     * @return Weight of the connection
     */
    public double getWeight(int id){
        return connectionWeight.get(id);
    }

    /**
     * getter for the output of a neuron (for testing)
     * @param id The id of the neuron.
     * @return The output of the neuron.
     */
    public double getOutput(int id){
        return neuronOutput.get(id);
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
        System.out.print("Epoch: " + epoch +"\n");
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


