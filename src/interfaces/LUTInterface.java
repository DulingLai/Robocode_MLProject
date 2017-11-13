package interfaces;

import java.util.List;

public interface LUTInterface extends CommonInterface {

    // Initialise the look up table to all zeros if learning; otherwise load the LUT from file
    public void initialiseLUT(boolean learning);

    /**
     * A helper method that translates a vector being used to index the look up table
     * into an ordinal that can then be used to access the associated look up table element.
     * @param X The state action vector used to index the LUT
     * @return The index where this vector maps to
     */
    public List<Integer> indexFor(double [] X); } // End of public interface LUT
