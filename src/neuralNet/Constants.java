package neuralNet;

public final class Constants {

    private Constants(){

    }


    /*
    Constants for XOR neural network (course work 1)
     */
    public static final int NUM_INPUTS = 9;
    public static final int NUM_OUTPUTS = 1;
    public static final int NUM_HIDDEN = 20;
    public static final double TARGET_ERROR_XOR = 0.05;
    public static final double LEARNING_RATE = 0.001;
    public static final double MOMENTUM = 0.9;
    public static final double ARG_A = -2.0;
    public static final double ARG_B = 2.0;
    public static final boolean BIPOLAR = false;
    public static final boolean CUSTOM_SIGMOID = false;

    public static final boolean IMMEDIATE_WEIGHT_UPDATE = true;
    public static final int MAX_EPOCHS = 10000;
    public static final int NUM_TRIAL = 1;
    public static final boolean SAVE_TO_FILE = true;

    // Text file path
    public static final String RESULT_FILE_PATH = "/Users/dulinglai/Documents/Study/CourseMaterials/GraduateCourse/EECE592/MLProject_Robocode/result/";
    public static final String RESULT_FILE_NAME = "BPTrainingResult";
}
