package com.josephcatrambone.aij;


import java.util.Random;
import java.text.DecimalFormat;

public class newBPTest
{
    private static final int INPUT_NEURONS = 2;
    private static final int HIDDEN_NEURONS = 4;
    private static final int OUTPUT_NEURONS = 1;

    private static final double LEARN_RATE = 0.2;    // Rho.
    private static final double NOISE_FACTOR = 0.45;
    private static final int TRAINING_REPS = 10000;

    // Input to Hidden Weights (with Biases).
    private static double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS];

    // Hidden to Output Weights (with Biases).
    private static double who[][] = new double[HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

    // Activations.
    private static double inputs[] = new double[INPUT_NEURONS];
    private static double hidden[] = new double[HIDDEN_NEURONS];
    private static double target[] = new double[OUTPUT_NEURONS];
    private static double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private static double erro[] = new double[OUTPUT_NEURONS];
    private static double errh[] = new double[HIDDEN_NEURONS];

    private static final int MAX_SAMPLES = 4;

//    private static int trainInputs[][] = new int[][] {{1, 1, 1, 0},
//            {1, 1, 0, 0},
//            {0, 1, 1, 0},
//            {1, 0, 1, 0},
//            {1, 0, 0, 0},
//            {0, 1, 0, 0},
//            {0, 0, 1, 0},
//            {1, 1, 1, 1},
//            {1, 1, 0, 1},
//            {0, 1, 1, 1},
//            {1, 0, 1, 1},
//            {1, 0, 0, 1},
//            {0, 1, 0, 1},
//            {0, 0, 1, 1}};

    private static int trainInputs[][] = new int[][] {{0,0}, {0,1}, {1,0}, {1,1}};

    private static int trainOutput[][] = new int[][] {{0},{1},{1},{0}};

    private static void NeuralNetwork()
    {
        int sample = 0;

        assignRandomWeights();

        // Train the network.
        for(int epoch = 0; epoch < TRAINING_REPS; epoch++)
        {
            sample += 1;
            if(sample == MAX_SAMPLES){
                sample = 0;
            }

            for(int i = 0; i < INPUT_NEURONS; i++)
            {
                inputs[i] = trainInputs[sample][i];
            } // i

            for(int i = 0; i < OUTPUT_NEURONS; i++)
            {
                target[i] = trainOutput[sample][i];
            } // i

            feedForward();

            backPropagate();

        } // epoch

        getTrainingStats();

//        System.out.println("\nTest network against original input:");
//        testNetworkTraining();
//
//        System.out.println("\nTest network against noisy input:");
//        testNetworkWithNoise1();

        return;
    }

    private static void getTrainingStats()
    {
        double sum = 0.0;
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j];
            } // j

            for(int j = 0; j < OUTPUT_NEURONS; j++)
            {
                target[j] = trainOutput[i][j];
            } // j

            feedForward();

            System.out.println(inputs[0] + "\t" + inputs[1]);
            System.out.println("Actual: " + actual[0] + "\t" + "Expected: " + target[0]);
        } // i

//        System.out.println("Network is " + ((double)sum / (double)MAX_SAMPLES * 100.0) + "% correct.");

        return;
    }

    private static void testNetworkTraining()
    {
        // This function simply tests the training vectors against network.
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j];
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static void testNetworkWithNoise1()
    {
        // This function adds a random fractional value to all the training
        // inputs greater than zero.
        DecimalFormat dfm = new java.text.DecimalFormat("###0.0");

        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j] + (new Random().nextDouble() * NOISE_FACTOR);
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(dfm.format(((inputs[j] * 1000.0) / 1000.0)) + "\t");
            } // j
            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static int maximum(final double[] vector)
    {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for(int index = 0; index < OUTPUT_NEURONS; index++)
        {
            if(vector[index] > max){
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }

    private static void feedForward()
    {
        double sum = 0.0;

        // Calculate input to hidden layer.
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * wih[inp][hid];
            } // inp

            sum += wih[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sigmoid(sum);
        } // hid

        // Calculate the hidden to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                sum += hidden[hid] * who[hid][out];
            } // hid

            sum += who[HIDDEN_NEURONS][out]; // Add in bias.
            actual[out] = sigmoid(sum);
        } // out
        return;
    }

    private static void backPropagate()
    {
        // Calculate the output layer error (step 3 for output cell).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
        }

        // Calculate the hidden layer error (step 3 for hidden cell).
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            errh[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh[hid] += erro[out] * who[hid][out];
            }
            errh[hid] *= sigmoidDerivative(hidden[hid]);
        }

        // Update the weights for the output layer (step 4).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                who[hid][out] += (LEARN_RATE * erro[out] * hidden[hid]);
            } // hid
            who[HIDDEN_NEURONS][out] += (LEARN_RATE * erro[out]); // Update the bias.
        } // out

        // Update the weights for the hidden layer (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                wih[inp][hid] += (LEARN_RATE * errh[hid] * inputs[inp]);
            } // inp
            wih[INPUT_NEURONS][hid] += (LEARN_RATE * errh[hid]); // Update the bias.
        } // hid
        return;
    }

    private static void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                wih[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp

        for(int hid = 0; hid <= HIDDEN_NEURONS; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                who[hid][out] = new Random().nextDouble() - 0.5;
            } // out
        } // hid
        return;
    }

    private static double sigmoid(final double val)
    {
        return (1.0 / (1.0 + Math.exp(-val)));
    }

    private static double sigmoidDerivative(final double val)
    {
        return (val * (1.0 - val));
    }

    public static void main(String[] args)
    {
        NeuralNetwork();
        return;
    }

}
