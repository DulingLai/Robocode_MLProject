package test;


import neuralNet.Constants;
import neuralNet.NeuralNet;
import org.junit.Before;
import org.junit.Test;

import java.io.FileWriter;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class testNeuralNet {
    NeuralNet nn_binary = new NeuralNet(
            2,    // int argNumInputs,
            4,    // int argNumHidden,
            1,  // int argNumOutputs,
            0.2,    // double argLearningRate,
            0.0,    // double argMomentumTerm,
            0.0,    // double argA,
            1.0,    // double argB,
            false);    // boolean argUseBipolarHIddenNeurons);

    NeuralNet nn_bipolar = new NeuralNet(
            2,    // int argNumInputs,
            4,    // int argNumHidden,
            1,  // int argNumOutputs,
            0.2,    // double argLearningRate,
            0.0,    // double argMomentumTerm,
            0.0,    // double argA,
            1.0,    // double argB,
            true);    // boolean argUseBipolarHIddenNeurons);

    @Before
    public void setUp() throws Exception {
    }

    // Test the sigmoid methods
    @Test
    public void testBinarySigmoid() {
        double x = 1;
        double expectedResult = 0.731058;
        double delta = 0.001;
        double actualResult = nn_binary.sigmoid(x);
        assertEquals(expectedResult, actualResult, delta);
    }

    @Test
    public void testBipolarSigmoid() {
        double x = 1;
        double expectedResult = 0.462117;
        double delta = 0.001;
        double actualResult = nn_bipolar.sigmoid(x);
        assertEquals(expectedResult, actualResult, delta);
    }

    @Test
    public void testSetAllWeights() {
        nn_binary.setAllWeights(0.2);
        for (int i = 1100; i < 1103; i++) {
            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
        }

        for (int i = 1200; i < 1203; i++) {
            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
        }

        for (int i = 1300; i < 1303; i++) {
            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
        }

        for (int i = 2110; i < 2115; i++) {
            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
        }


    }

    @Test
    public void testFeedForward() {
        nn_binary.setAllWeights(0.0);
        assertEquals(0.5, nn_binary.outputFor(Constants.INPUTS_BINARY[0]), 0.00001);

        nn_binary.setAllWeights(0.2);
        assertEquals(0.6718, nn_binary.outputFor(Constants.INPUTS_BINARY[0]), 0.0001);
        assertEquals(0.6635, nn_binary.outputFor(Constants.INPUTS_BINARY[1]), 0.0001);
        assertEquals(0.6635, nn_binary.outputFor(Constants.INPUTS_BINARY[2]), 0.0001);
        assertEquals(0.6547, nn_binary.outputFor(Constants.INPUTS_BINARY[3]), 0.0001);
    }

    @Test
    public void testCourseWork_1() {
        for(int i=1; i<Constants.NUM_TRIAL+1;i++){
            String fileName = Constants.RESULT_FILE_NAME +"_" + i + ".txt";
            int num_epoch = runBPTraining(fileName);

            // print the distribution of all test runs
            try {
                FileWriter fw = new FileWriter(Constants.RESULT_FILE_PATH+"BP_aggregate_test_result.txt", true);
                fw.write(i + ", " + num_epoch + "\n");
                fw.flush();
                fw.close();
            } catch(IOException e){
                System.out.println("Error reading the aggregate text file! Please check the file path in code!");
            }
        }

    }

    private int runBPTraining(String fileName) {
        // instantiate the neural net
        NeuralNet nn = new NeuralNet(
                Constants.NUM_INPUTS,
                Constants.NUM_HIDDEN,
                Constants.NUM_OUTPUTS,
                Constants.LEARNING_RATE,
                Constants.MOMENTUM,
                Constants.ARG_A,
                Constants.ARG_B,
                Constants.BIPOLAR);

        // Variables
        int max_epoch = Constants.MAX_EPOCHS;
        double target_error = Constants.TARGET_ERROR_XOR;
        boolean saveToFile = Constants.SAVE_TO_FILE;

        double expectedResult[];
        double inputs[][];
        double actualResult[] = {0, 0, 0, 0};

        if (Constants.BIPOLAR) {
            inputs = Constants.INPUTS_BIPOLAR;
            expectedResult = Constants.EXPECT_OUTPUT_BIPOLAR;
        } else {
            inputs = Constants.INPUTS_BINARY;
            expectedResult = Constants.EXPECT_OUTPUT_BINARY;
        }


        // initialize random weights for all connections
        nn.initializeWeights();

        // initialize error term
        double error = 2.0;

        // initialize the number of epoch
        int num_epoch = 1;

        while (num_epoch <= max_epoch && error > target_error) {
            // clear error for each epoch
            error = 0;

            // train for each epoch: 4 pairs of inputs and outputs
            for (int j = 0; j < inputs.length; j++) {
                actualResult[j] = nn.train(inputs[j], expectedResult[j]);
                error += 0.5 * Math.pow(expectedResult[j] - actualResult[j], 2);
            }

            // print the result for each epoch
            nn.printTrainResult(inputs, expectedResult, actualResult, error, num_epoch, saveToFile, fileName);
            num_epoch++;
        }

        return num_epoch;
    }
}

