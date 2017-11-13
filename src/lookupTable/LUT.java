package lookupTable;

import interfaces.LUTInterface;

import java.io.*;
import java.util.*;

public class LUT implements LUTInterface{

    private int argNumInputs;
    private int[] argVariableFloor;
    private int [] argVariableCeiling;

    private int dimensionality = 1;

    // lookup table
    private HashMap<List<Integer>, Double> lookupTable = new HashMap<>();

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

    public HashMap<List<Integer>, Double> getLookupTable() {
        return lookupTable;
    }

    @Override
    public void initialiseLUT(boolean learning) {
//        for(int i=0; i<argNumInputs; i++){
//            if(argVariableFloor[i] == 0){
//                dimensionality = dimensionality * 2;
//            }else {
//                dimensionality = dimensionality * (argVariableCeiling[i] - argVariableFloor[i]);
//            }
//        }
//
//        // print the actual number of total states of the LUT initialized
//        dimensionality = dimensionality * 5; // 5 refers to the number of actions
//        System.out.println("The dimensionality of the LUT is: " + dimensionality);

        if(!learning){
            try {
                load("lookupTable.txt");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public List<Integer> indexFor(double[] X) {
        List<Integer> index = new ArrayList<>();
        for(int i = 0; i< X.length; i++){
            index.add((int)X[i]);
        }
        return index;
    }

    @Override
    public double outputFor(double[] X) {
        List<Integer> i = indexFor(X);
        lookupTable.putIfAbsent(i, 0.0);
        return lookupTable.get(i);
    }

    @Override
    public double train(double[] X, double argValue) {
        List<Integer> i = indexFor(X);
        lookupTable.put(i, argValue);
        return -Double.NEGATIVE_INFINITY;
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

            for(Map.Entry<List<Integer>, Double> entry : lookupTable.entrySet()){
                for(Integer i : entry.getKey()) {
                    writeLUT.writeInt(i);
                }
                writeLUT.writeDouble(entry.getValue());
                writeLUT.writeBytes(newLine);
            }
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
