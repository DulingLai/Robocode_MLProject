package test;

import lookupTable.LUT;

import neuralNet.Constants;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.neural.networks.training.propagation.sgd.update.MomentumUpdate;
import org.encog.util.Format;
import org.encog.util.simple.EncogUtility;
import org.junit.Before;
import org.junit.Test;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class RL_NN_test {

    private static double[][] inputs;
    private static double[][] expectedResult;

    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void testLUT(){

        int numInput = 9;
        int numHidden = 17;
        int numOutput = 1;
        String fileName = Constants.RESULT_FILE_NAME + ".txt";

        try {
            loadLUT();
        } catch (IOException e) {
            e.printStackTrace();
        }

//        for(int trial = 6; trial<40; trial++) {
//
//            numHidden = trial;
            // setup nn
            BasicNetwork network = new BasicNetwork();
            network.addLayer(new BasicLayer(null, true, numInput));
            network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHidden));
            network.addLayer(new BasicLayer(new ActivationSigmoid(), false, numOutput));
            network.getStructure().finalizeStructure();
            network.reset();

            // Create training data.
            MLDataSet trainingSet = new BasicMLDataSet(inputs, expectedResult);

            // Train the neural network. (10-fold cross validation is used)
//            final FoldedDataSet foldedDataSet = new FoldedDataSet(trainingSet);             // create folded dataset
//            final MLTrain train = new ResilientPropagation(network, foldedDataSet);
//            final CrossValidationKFold trainFolded = new CrossValidationKFold(train, 4);   // 4-fold


            // train the neural network (no cross validation)
//            final ResilientPropagation trainFolded = new ResilientPropagation(network, trainingSet);
//
//            int epoch = 1;
//
//            do {
//                trainFolded.iteration();
//                printTrainResult(numHidden, trainFolded.getError(), epoch, true, fileName);
//                epoch++;
//            } while (trainFolded.getError() > 0.09 && epoch < 5000);

//        trainFolded.finishTraining();

        /*
        Online training
         */
        new XaiverRandomizer(42).randomize(network);
        final StochasticGradientDescent sgd = new StochasticGradientDescent(network, trainingSet);
        sgd.setLearningRate(0.05);
        sgd.setMomentum(0.5);
        sgd.setUpdateRule(new MomentumUpdate());

        double error = Double.POSITIVE_INFINITY;
        int epoch = 1;
        int sample = expectedResult.length;

        while(epoch<2000) {
            for (int i = 0; i < sample; i++) {
                MLDataPair pair = trainingSet.get(i);

                // Update the gradients based on this pair.
                sgd.process(pair);
                // Update the weights, based on the gradients
                sgd.update();
            }

            // Calculate the overall error.  You might not want to do this every step on a large data set.
            error = network.calculateError(trainingSet);

            printTrainResult(numHidden, error, epoch, true, fileName);
            epoch++;
        }

//            double trainError = trainFolded.getError();

            // save results
//            printTrainResult(numHidden, trainError, epoch, true, fileName);
//        }

        // Shut down Encog.
        Encog.getInstance().shutdown();
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
        expectedResult = new double[arraySize][1];

        int index = 0;
        for (Map.Entry<String, Double> entry : LUT.entrySet()){
            double[] X = new double[9];
            String[] stringArray = entry.getKey().split(",");
            for (int i = 0; i < 9; i++){
                X[i] = Double.parseDouble(stringArray[i]);
            }
            inputs[index] = X;
            expectedResult[index][0] = entry.getValue();
            index++;
        }
    }

    /**
     * print the training result to the console or a txt file
     * @param error
     * @param epoch
     * @param saveToFile
     */
    public void printTrainResult(int numHidden,double error, int epoch, boolean saveToFile, String fileName) {

        // print the training result to the console
        System.out.print("Hidden Neuron: " + numHidden + ", ");
        System.out.print("Epoch: " + epoch + ", ");
        System.out.println("ERROR: " + error + "; ");

        // print the training result to a txt file
        if (saveToFile) {
            try {
                FileWriter fw = new FileWriter(Constants.RESULT_FILE_PATH + fileName, true);
                fw.write(numHidden + ", " + epoch + ", " + error + "\n");
                fw.flush();
                fw.close();
            } catch (IOException e) {
                System.out.println("Error reading text file! Please check the file path in code!");
            }
        }
    }
}
