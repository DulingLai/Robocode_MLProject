package test;


import lookupTable.LUT;
import neuralNet.Constants;
import neuralNet.NeuralNet;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class testNeuralNet {

    private static double[][] inputs;
    private static double[] expectedResult;

//    NeuralNet nn_binary = new NeuralNet(
//            Constants.NUM_INPUTS,    // int argNumInputs,
//            Constants.NUM_HIDDEN,    // int argNumHidden,
//            Constants.NUM_OUTPUTS,  // int argNumOutputs,
//            Constants.LEARNING_RATE,    // double argLearningRate,
//            Constants.MOMENTUM,    // double argMomentumTerm,
//            Constants.ARG_A,    // double argA,
//            Constants.ARG_B,    // double argB,
//            false);    // boolean argUseBipolarHIddenNeurons);
//
//    NeuralNet nn_bipolar = new NeuralNet(
//            6,    // int argNumInputs,
//            30,    // int argNumHidden,
//            1,  // int argNumOutputs,
//            0.2,    // double argLearningRate,
//            0.0,    // double argMomentumTerm,
//            0.0,    // double argA,
//            1.0,    // double argB,
//            true);    // boolean argUseBipolarHIddenNeurons);

    @Before
    public void setUp() throws Exception {
    }

//    // Test the sigmoid methods
    @Test
    public void testBinarySigmoid() {
        // initialize neural network
        NeuralNet nn_binary = new NeuralNet(
                2,    // int argNumInputs,
                4,    // int argNumHidden,
                1,  // int argNumOutputs,
                0.2,    // double argLearningRate,
                0.9,    // double argMomentumTerm,
                Constants.ARG_A,    // double argA,
                Constants.ARG_B,    // double argB,
                false);

        double x = 1;
        double expectedResult = 0.731058;
        double delta = 0.001;
        double actualResult = nn_binary.sigmoid(x);
        assertEquals(expectedResult, actualResult, delta);
    }

    @Test
    public void testBipolarSigmoid() {
            NeuralNet nn_bipolar = new NeuralNet(
            2,    // int argNumInputs,
            4,    // int argNumHidden,
            1,  // int argNumOutputs,
            0.2,    // double argLearningRate,
            0.9,    // double argMomentumTerm,
            0.0,    // double argA,
            1.0,    // double argB,
            true);    // boolean argUseBipolarHIddenNeurons);

        double x = 1;
        double expectedResult = 0.462117;
        double delta = 0.001;
        double actualResult = nn_bipolar.sigmoid(x);
        assertEquals(expectedResult, actualResult, delta);
    }

//    @Test
//    public void testSetAllWeights() {
//        nn_binary.setAllWeights(0.2);
//        for (int i = 1100; i < 1103; i++) {
//            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
//        }
//
//        for (int i = 1200; i < 1203; i++) {
//            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
//        }
//
//        for (int i = 1300; i < 1303; i++) {
//            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
//        }
//
//        for (int i = 2110; i < 2115; i++) {
//            assertEquals(0.2, nn_binary.getWeight(i), 0.00001);
//        }
//    }

    @Test
    public void testFeedForward() {
        // initialize neural network
        NeuralNet nn_binary = new NeuralNet(
                2,    // int argNumInputs,
                4,    // int argNumHidden,
                1,  // int argNumOutputs,
                0.2,    // double argLearningRate,
                0.9,    // double argMomentumTerm,
                Constants.ARG_A,    // double argA,
                Constants.ARG_B,    // double argB,
                false);

        double[][] input_binary = {{0, 0}, {0, 1}, {1,0}, {1,1}};

        nn_binary.setAllWeights(0.0);
        assertEquals(0.5, nn_binary.outputFor(input_binary[0]), 0.00001);

        nn_binary.setAllWeights(0.2);
        assertEquals(0.65472344, nn_binary.outputFor(input_binary[0]), 0.0001);
        assertEquals(0.66350434, nn_binary.outputFor(input_binary[1]), 0.0001);
        assertEquals(0.66350434, nn_binary.outputFor(input_binary[2]), 0.0001);
        assertEquals(0.67184135, nn_binary.outputFor(input_binary[3]), 0.0001);
    }

    @Test
    public void testCourseWork_1() {
        // initialize neural network
        NeuralNet nn_binary = new NeuralNet(
            2,    // int argNumInputs,
            4,    // int argNumHidden,
            1,  // int argNumOutputs,
            0.02,    // double argLearningRate,
            0.9,    // double argMomentumTerm,
            Constants.ARG_A,    // double argA,
            Constants.ARG_B,    // double argB,
            false);    // boolean argUseBipolarHIddenNeurons);

        double[][] input_binary = {{0, 0}, {0, 1}, {1,0}, {1,1}};
        double[] expected_binary = {0, 1, 1, 0};

        String fileName = Constants.RESULT_FILE_NAME + ".txt";
        nn_binary.initializeWeights();
        nn_binary.setAllWeights(0.2);

        int epoch = 0;
        double error = 2.0;
        while (epoch <= Constants.MAX_EPOCHS && error > Constants.TARGET_ERROR_XOR) {
            // clear error for each epoch
            error = 0;
            double[] actual_result = new double[expected_binary.length];

            // train for each epoch: 19000+ pairs of inputs and outputs
            for (int j = 0; j < expected_binary.length; j++) {
                actual_result[j] = nn_binary.train(input_binary[j], expected_binary[j]);
                error += Math.pow(expected_binary[j] - actual_result[j], 2);
            }

            // print the result for each epoch
            nn_binary.printTrainResult(input_binary, expected_binary, actual_result, error, epoch, true, fileName);
            epoch++;
        }
    }

//    @Test
//    public void testCourseWork_3() {
//        int hiddenNeuron = 10;
//        for(int i=1; i<Constants.NUM_TRIAL+1;i++){
//            String fileName = Constants.RESULT_FILE_NAME +"_" + i + ".txt";
//            int num_epoch = runBPTraining(fileName);
//
//            // print the distribution of all test runs
//            try {
//                FileWriter fw = new FileWriter(Constants.RESULT_FILE_PATH+"BP_aggregate_test_result.txt", true);
//                fw.write(i + ", " + num_epoch + "\n");
//                fw.flush();
//                fw.close();
//            } catch(IOException e){
//                System.out.println("Error reading the aggregate text file! Please check the file path in code!");
//            }
//        }
//    }

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

        // we load the LUT to inputs[][] and expectedResult[]
        try {
            loadLUT();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double actualResult[] = new double[expectedResult.length];

        // initialize random weights for all connections
        nn.initializeWeights();

        // initialize error term
        double error = 40000;

        // initialize the number of epoch
        int num_epoch = 1;

        while (num_epoch <= max_epoch && error > target_error) {
            // clear error for each epoch
            error = 0;

            // train for each epoch: 19000+ pairs of inputs and outputs
            for (int j = 0; j < expectedResult.length; j++) {
                actualResult[j] = nn.train(inputs[j], expectedResult[j]);
                error += Math.pow(expectedResult[j] - actualResult[j], 2);
            }

            error = Math.sqrt(error/expectedResult.length);

            // print the result for each epoch
            nn.printTrainResult(inputs, expectedResult, actualResult, error, num_epoch, saveToFile, fileName);
            num_epoch++;
        }

        return num_epoch;
    }

    private void loadLUT() throws IOException {
        double arenaWidth = 800.0;
        double arenaHeight = 600.0;
        double scalingFactor = 0.01;
        /*
        Initialize the instance of look up table
        */
        final int[] floors = {
                (int) (-arenaWidth * scalingFactor / 2),
                (int) (-arenaHeight * scalingFactor / 2),
                (int) (-arenaWidth * scalingFactor / 2),
                (int) (-arenaHeight * scalingFactor / 2),
                0,
                0, 0, 0, 0   // lower bound for actions
        };

        final int[] ceilings = {
                (int) (+arenaWidth * scalingFactor / 2),
                (int) (+arenaHeight * scalingFactor / 2),
                (int) (+arenaWidth * scalingFactor / 2),
                (int) (+arenaHeight * scalingFactor / 2),
                0,
                1, 1, 1, 1   // upper bound for actions
        };

        LUT myLUT = new LUT(9, floors, ceilings);

        // Input stream to read in the lookup table values
        DataInputStream scanner = new DataInputStream(new FileInputStream(
                "/Users/dulinglai/Documents/Study/CourseMaterials/GraduateCourse/EECE592/MLProject_Robocode/out/production/Robocode_MLProject/bots/BasicWaveSurferBot.data/LUT.dat"));
        String argString;
        Double argValue;
        HashMap<String, Double> LUT = myLUT.getLookupTable();

        while(scanner.available()>0) {
            // read the lookup table values
            argString = scanner.readUTF();
            argValue = scanner.readDouble();


            LUT.putIfAbsent(argString, argValue);
        }
        scanner.close();

        // initialize inputs and expected results
        int arraySize = LUT.size();
        inputs = new double[arraySize][9];
        expectedResult = new double[arraySize];

        int index = 0;
        for (Map.Entry<String, Double> entry : LUT.entrySet()){
            double[] X = new double[9];
            String[] stringArray = entry.getKey().split(",");
            for (int i = 0; i < 9; i++){
                X[i] = Double.parseDouble(stringArray[i]);
            }
            inputs[index] = X;
            expectedResult[index] = entry.getValue();
            index++;
        }
    }
}

