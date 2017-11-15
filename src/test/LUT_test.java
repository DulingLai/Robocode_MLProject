package test;

import lookupTable.LUT;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LUT_test {

    private static final int NUM_STATES = 6;
    private static final int arenaWidth = 800;
    private static final int arenaHeight = 600;
    private static final double scalingFactor = 0.02;

    private final static int[] floors = {
            (int) (-arenaWidth * scalingFactor),
            (int) (-arenaHeight * scalingFactor),
            (int) (-arenaWidth * scalingFactor / 2),
            (int) (-arenaHeight * scalingFactor / 2),
            0   // lower bound for actions
    };

    private final static int[] ceilings = {
            (int) (+arenaWidth * scalingFactor),
            (int) (+arenaHeight * scalingFactor),
            (int) (+arenaWidth * scalingFactor / 2),
            (int) (+arenaHeight * scalingFactor / 2),
            4   // upper bound for actions
    };

    private static LUT myLUT = new LUT(NUM_STATES + 1, floors, ceilings);

    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void testTrainAndLoad() {
        double[] X = new double[]{-1, 1, 2, 5, -3, 0, 4};
        String expectedIndex = "-1125-304";
        double argValue = -2.1111889;
        myLUT.train(X, argValue);
        assertEquals(expectedIndex, myLUT.indexFor(X));
        assertEquals(argValue, myLUT.outputFor(X), 0.00001);
    }
}
