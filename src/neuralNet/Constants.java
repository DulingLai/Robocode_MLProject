package neuralNet;

public final class Constants {

    private Constants(){

    }


    /*
    Constants for XOR neural network (course work 1)
     */
    public static final int NUM_INPUTS = 2;
    public static final int NUM_OUTPUTS = 1;
    public static final int NUM_HIDDEN = 4;
    public static final double TARGET_ERROR_XOR = 0.05;
    public static final double LEARNING_RATE = 0.2;
    public static final double MOMENTUM = 0.9;
    public static final double ARG_A = 0.0;
    public static final double ARG_B = 0.0;
    public static final boolean BIPOLAR = false;

    public static final double WEIGHT_MULTIPLIER = 0.5;
    public static final boolean IMMEDIATE_WEIGHT_UPDATE = true;
    public static final int MAX_EPOCHS = 10000;
    public static final int NUM_TRIAL = 100;
    public static final boolean SAVE_TO_FILE = true;

    public static final double INPUTS_BINARY[][] = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};
    public static final double INPUTS_BIPOLAR[][] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    public static final double EXPECT_OUTPUT_BINARY[] = {0, 1, 1, 0};
    public static final double EXPECT_OUTPUT_BIPOLAR[] = {-1, 1, 1, -1};

    // Text file path
    public static final String RESULT_FILE_PATH = "/Users/dulinglai/Documents/Study/CourseMaterials/GraduateCourse/EECE592/CourseWork_1/result/";
    public static final String RESULT_FILE_NAME = "BPTrainingResult";
}
