package lookupTable;

import interfaces.LUTInterface;

import java.io.*;
import java.util.*;

public class LUT implements LUTInterface{

    private int argNumInputs;
    private int[] argVariableFloor;
    private int [] argVariableCeiling;

    // lookup table
    private HashMap<String, Double> lookupTable = new HashMap<>();

    /**
     * Constructor. (You will need to define one in your implementation)
     * @param argNumInputs The number of inputs in your input vector
     * @param argVariableFloor An array specifying the lowest value of each variable in the input vector.
     * @param argVariableCeiling An array specifying the highest value of each of the variables in the input vector.
     * The order must match the order as referred to in argVariableFloor. *
     **/
    public LUT (
        int argNumInputs,
        int [] argVariableFloor,
        int [] argVariableCeiling )
    {
        this.argNumInputs = argNumInputs;
        this.argVariableFloor = argVariableFloor;
        this.argVariableCeiling = argVariableCeiling;
    }

    public HashMap<String, Double> getLookupTable() {
        return lookupTable;
    }

    @Override
    public void initialiseLUT() {
        try {
            load("LUT.dat");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String indexFor(double[] X) {
        String index = "";
        for(int i = 0; i< X.length; i++){
            index += String.valueOf((int)X[i]) + ",";
        }
        return index;
    }

    @Override
    public double outputFor(double[] X) {
        String i = indexFor(X);
        if(lookupTable.containsKey(i)){
            System.out.println("Found a match state in the LUT! ");
            return lookupTable.get(i);
        } else return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        String i = indexFor(X);
        lookupTable.put(i, argValue);
        return 0;
    }

    public double[] decodeKey(String key){
        double[] X = new double[argNumInputs];
        char[] charArray = key.toCharArray();
        for (int i = 0; i < argNumInputs; i++){
            X[i] = charArray[i];
        }
        return X;
    }

    @Override
    public void save(File argFile) {
        // create the file if it does not exist
        File lutFile = argFile;
        String newLine = System.getProperty("line.separator");
        try{
            lutFile.createNewFile();
        } catch(IOException e){
            System.out.println("The Lookup Table file does not exist!");
        }

        // write to the file
        try {
            DataOutputStream writeLUT = new DataOutputStream(new FileOutputStream(lutFile, false));

//            for(Map.Entry<String, Double> entry : lookupTable.entrySet()){
//                for(String i : entry.getKey()) {
//                    writeLUT.writeInt(i);
//                }
//                writeLUT.writeDouble(entry.getValue());
//                writeLUT.writeBytes(newLine);
//            }
            writeLUT.close();

        } catch(IOException e){

        }
    }

    @Override
    public void load(String argFileName) throws IOException {
        // use a scanner to scan through the files for lookup table values
        File lutFile = new File("RL_robot.data/lookupTable.txt");
        Scanner scan = new Scanner(lutFile);
        double[] key = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // read the lookup table values
        while(scan.hasNextDouble()){
            for(int i=0; i<argNumInputs; i++){
                key[i]=scan.nextDouble();
            }
            key[10] = scan.nextDouble();
        }
        scan.close();
    }
}
