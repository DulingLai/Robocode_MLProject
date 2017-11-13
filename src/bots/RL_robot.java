package bots;

import lookupTable.LUT;
import robocode.*;
import robocode.util.Utils;

import java.awt.Color;
import java.io.*;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static robocode.util.Utils.normalRelativeAngleDegrees;

public class RL_robot extends AdvancedRobot{

    private static final boolean debug = true;

    /*
    ===============================    LUT related variables    =============================
     */

    /*
     state-action space
     States: deltaX, deltaY from enemy; deltaX, deltaY from center, enemy low/ok (-1/+1)
     Actions: move up/down/left/right; fire
      */

    private static double[][] stateActionTable = {
            {0, 0, 0, 0, 0, 0},             // pre-populated for action moveUp
            {0, 0, 0, 0, 0, 1},             // pre-populated for action moveDown
            {0, 0, 0, 0, 0, 2},             // pre-populated for action moveLeft
            {0, 0, 0, 0, 0, 3},             // pre-populated for action moveRight
            {0, 0, 0, 0, 0, 4}             // pre-populated for action fire
    };

    // Constants for LUT State actions
    private static final int NUM_STATES = 5;
    private static final int NUM_ACTIONS = 1;


    // Discount factor and learning rate
    private static final double GAMMA = 0.9;
    private static final double ALPHA = 0.7;

    // previous and current Q value
    private static double currentQ = 0.0;
    private static double previousQ = 0.0;
    private static double errorQ = 0.0;

    // Rewards weights
    private static final double badInstantReward = -0.25;
    private static final double badTerminalReward = -0.5;
    private static final double goodInstantReward = 0.025;
    private static final double goodTerminalReward = 0.75;

    // Control exploring and learning
    private static boolean learning = true;
    private static boolean exploring = false;
    private static boolean onPolicy = false;
    private static boolean enableInstantRewards = true;
    private static final double epsilon = 0.05; // % exploration when turned on

    /*
    Statistics of learning
     */
    private final int AVERAGE_SAMPLE_SIZE = 1000; // number of rounds which average is calculated
    private static int sampleCount = 0;
    private static int numWins = 0;

    // Total accumulated rewards
    private static double accumulatedRewards = 0.0;
    private boolean instantReward = true;

    private static int numBackSteps = 0;
    private static double averageSumQ = 0.0;
    private static double averageErrorQ = 0.0;

    // event callbacks
    private static int numWallHit = 0;
    private static int numBulletHit = 0;
    private static int numHitByBullet = 0;

    private static double totalNNSquaredError = 0;
    private static int winForAutosaving = 100; // automatically save the weights after number of wins

    /*
    Formatting log files
     */
    DecimalFormat dfR = new DecimalFormat("#0.00");
    DecimalFormat dfE = new DecimalFormat("0.000");
    DecimalFormat dfW = new DecimalFormat("000");


    /*
    ===============================    Robot control related variables    =============================
     */

    // Threshold for low energy
    private static final double lowEnergy = 40;

//    // if we have locked on target
//    private static boolean enemyLockedOn = false;

    // Gun power used: 1,2,3
    private static final int gunPower = 1;

    // Distance moved each step
    private static final int stepDistance = 50;

    // x,y scaling factor (for quantization); for example, if scalingFactor = 0.01: x_states = 800 * 0.01 = 8;
    private static final double scalingFactor = 0.02;
    // constants for arena dimension
    private static final double arenaWidth = 800;
    private static final double arenaHeight = 600;

    // Enum to index the action array
    private enum RobotActions {UP, DOWN, LEFT, RIGHT, FIRE}
    private RobotActions selectedAction;

    // initialize current and previous state action
    private static double[] currentStateAction = new double[]{0, 0, 0, 0, 0, 0};
    private static double[] previousStateAction = new double[]{0, 0, 0, 0, 0, 0};


    // Enum of the state machine of the Robot, it first scan to get a sense of the current state s',
    // then select an action to maximize Q(s',a'), then it performs the action,
    // wait for the reward event when it updates the previous Q(s,a) and store s <- s', go back to scan mode
    private enum RobotMode {SCAN, IDLE}
    private RobotMode mode;


    // store the enemy tank status
//    private double bearing = 0.0;
//    private double distance = 0.0;
//    private double energy = 0.0;
//    private double heading = 0.0;
//    private double speed = 0.0;
//    private boolean isEnemyStatusUpToDate = false;

    private boolean closeToTopWall = false, closeToLeftWall = false, closeToRightWall = false, closeToBottomWall = false;



    /*
    Initialize the instance of look up table
     */
    private final static int [] floors = {
            (int)(-arenaWidth * scalingFactor),
            (int)(-arenaHeight * scalingFactor),
            (int)(-arenaWidth * scalingFactor / 2),
            (int)(-arenaHeight * scalingFactor / 2),
            0   // lower bound for actions
    };

    private final static int [] ceilings = {
            (int)(+arenaWidth * scalingFactor),
            (int)(+arenaHeight * scalingFactor),
            (int)(+arenaWidth * scalingFactor / 2),
            (int)(+arenaHeight * scalingFactor / 2),
            4   // upper bound for actions
    };

    private LUT myLUT = new LUT(NUM_STATES+NUM_ACTIONS, floors, ceilings);


    // the Robocode main method
    @Override
    public void run() {
        // check the dimension of the arena, if it is not equal to the preset constants, raise an exception
        if (getBattleFieldHeight() != arenaHeight || getBattleFieldWidth() != arenaWidth) {
            throw new IllegalArgumentException("The actual battle field dimension is: " + getBattleFieldWidth() + " x " + getBattleFieldHeight());
        }

        /*
         Customize the robot
          */
        // Make robot, gun, adn radar turn independently of each other
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setBulletColor(Color.red);
        if (learning) {
            if (exploring) setGunColor(Color.yellow);
            else setGunColor(Color.red);
        } else setGunColor(Color.BLACK);

        // initialize the state machine to scan mode
        mode = RobotMode.SCAN;

        // initialize the action to going up
        selectedAction = RobotActions.UP;

        // initialize the LUT
        myLUT.initialiseLUT(learning);

        // Here we use Turn Multiplier Lock to ensure a lock on the enemy
        turnRadarRightRadians(Double.POSITIVE_INFINITY);
        do {
            scan();
        } while (true);
    }


    private void updateStateAction() {
        // initialize max Q to -infinity
        double maxQ = Double.NEGATIVE_INFINITY;

        // check if the robot is close to the wall: if it is, set the flags accordingly
        checkCloseToWall();

        // Get Q value for all (state, action) pairs for current state
        for (RobotActions action : RobotActions.values()) {
            // Use the LUT to find the Q values
            // select the action with max Q
            switch (action) {
                case FIRE:
                    maxQ = checkForMaxQ(maxQ, action);
                    break;

                case LEFT:
                    maxQ = checkForMaxQ(maxQ, action);
                    break;

                case RIGHT:
                    maxQ = checkForMaxQ(maxQ, action);
                    break;

                case UP:
                    maxQ = checkForMaxQ(maxQ, action);
                    break;

                case DOWN:
                    maxQ = checkForMaxQ(maxQ, action);
                    break;
            }
        }

        // The following will override the selected action if exploring is turned on
        if (exploring) {
            Random rand = new Random();
            // epsilon greedy policy
            if (rand.nextDouble() < epsilon) {
                int i = rand.nextInt(NUM_ACTIONS);  //generates a random number between 0 and NUM_ACTIONS-1
                currentQ = myLUT.outputFor(stateActionTable[i]);
                for(int j=0; j< NUM_STATES+NUM_ACTIONS; j++) {
                    currentStateAction[j] = stateActionTable[i][j];
                }
                selectedAction = RobotActions.values()[i];
            }
        } else { // go with the best Q
            currentQ = maxQ;
            for(int i=0; i< NUM_STATES+NUM_ACTIONS; i++) {
                currentStateAction[i] = stateActionTable[selectedAction.ordinal()][i];
            }
        }

        // add up the total Q
        averageSumQ += currentQ;
    }

    private void performAction(ScannedRobotEvent enemyRobot){
        double currentHeading = getHeading();

        // detect enemy energy states (if enemy has 0 energy, kill it)
        double enemyEnergy = enemyRobot.getEnergy();

        // variable control how we want to turn the gun
        double gunTurnAmount = normalRelativeAngleDegrees(enemyRobot.getBearing() + getHeading() - getGunHeading());

        // when the enemy has no energy left, we kill it by rapid fire
        while(enemyEnergy == 0){
            turnGunRight(gunTurnAmount);
            fire(3);
        }

        // perform the selected action
        switch (selectedAction) {
            case UP:    // change heading to 0 (up)
                if (currentHeading > 180) {
                    setTurnRight(360 - currentHeading);
                } else {
                    setTurnLeft(currentHeading);
                }
                move(closeToTopWall);
                break;

            case DOWN:  // change heading to 180 (down)
                if (currentHeading < 180) {
                    setTurnRight(180 - currentHeading);
                } else {
                    setTurnLeft(currentHeading - 180);
                }
                move(closeToBottomWall);
                break;

            case LEFT:  // change heading to 270 (left)
                if (currentHeading < 270) {
                    setTurnRight(270 - currentHeading);
                } else {
                    setTurnLeft(currentHeading - 270);
                }
                move(closeToLeftWall);
                break;

            case RIGHT:  // change heading to 90 (right)
                if (currentHeading < 90) {
                    setTurnRight(90 - currentHeading);
                } else {
                    setTurnLeft(currentHeading - 90);
                }
                move(closeToRightWall);
                break;

            case FIRE:  // aim at the energy and fire
                setTurnGunRight(gunTurnAmount);
                setFire(gunPower);
                break;

        }
        execute();
    }

    private void updateStateActionTable(ScannedRobotEvent enemyRobot) {
        // read enemy & own robot status for states
        double deltaX = Math.sin(enemyRobot.getBearingRadians() + getHeadingRadians()) * enemyRobot.getDistance() * scalingFactor;
        double deltaY = Math.cos(enemyRobot.getBearingRadians() + getHeadingRadians()) * enemyRobot.getDistance() * scalingFactor;
        double xFromCenter = (getX() - arenaWidth/2.0) * scalingFactor;
        double yFromCenter = (getY() - arenaHeight/2.0) * scalingFactor;
        double ownEnergy = (this.getEnergy() >= lowEnergy)? +1 : -1; // boolean states: own energy is -1 if less than low energy threshold

        // update the state action table
        for(RobotActions i: RobotActions.values()){
            stateActionTable[i.ordinal()][0] = (double)Math.round(deltaX);
            stateActionTable[i.ordinal()][1] = (double)Math.round(deltaY);
            stateActionTable[i.ordinal()][2] = (double)Math.round(xFromCenter);
            stateActionTable[i.ordinal()][3] = (double)Math.round(yFromCenter);
            stateActionTable[i.ordinal()][4] = ownEnergy;
        }
    }

    private double checkForMaxQ(double maxQ, RobotActions action) {
        double tempQ;
        Random rand = new Random();
        tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
        if(tempQ == maxQ){
            if(rand.nextBoolean()){
                selectedAction = action;
            }
        } else if(tempQ > maxQ){
            maxQ = tempQ;
            selectedAction = action;
        }

        return maxQ;
    }

    private void checkCloseToWall() {
        // Get the position of our tank
        double x = getX();
        double y = getY();

        // determine if our tank is close to wall
        closeToBottomWall = false;
        closeToTopWall = false;
        closeToLeftWall = false;
        closeToRightWall = false;

        if(x<100){
            closeToLeftWall = true;
        } else if (x>700){
            closeToRightWall = true;
        }

        if(y<100){
            closeToBottomWall = true;
        } else if (y>500){
            closeToTopWall = true;
        }
    }

    private void move(boolean closeToWall) {
        if(closeToWall){
            setBack(stepDistance);
        } else {
            setAhead(stepDistance);
        }
    }

    /*
    callbacks: onScannedRobot, onHitByBullet, onBulletHit, onDeath, onWin
     */

    // This is the main callback where we update the states, select action and then perform action
    @Override
    public void onScannedRobot(ScannedRobotEvent enemyRobot) {

        // Turn Multiplier Lock
        double radarTurn = getHeadingRadians() + enemyRobot.getBearingRadians() - getRadarHeadingRadians();
        turnRadarRightRadians(2.0 * Utils.normalRelativeAngle(radarTurn));

        // We only update the states and take actions when we are in scanning mode
        if(mode == RobotMode.SCAN) {
            // update the current state action table
            updateStateActionTable(enemyRobot);

            // select action based on Q(s,a)
            updateStateAction();

            // perform actions
            performAction(enemyRobot);

            // change the mode to IDLE and wait for rewards to evaluate our next action
            mode = RobotMode.IDLE;
        }
    }

    /*
    when we hit by a bullet, assign a negative reward and do a back step
     */
    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        backStep(badInstantReward);
        numHitByBullet++;
    }

    /*
    when we hit the wall
     */
    @Override
    public void onHitWall(HitWallEvent event) {
        backStep(badInstantReward);
        numWallHit++;
        // we step back from the wall
        back(stepDistance+50);
    }

    /*
    when we hit the enemy
     */

    @Override
    public void onHitRobot(HitRobotEvent event) {
        backStep(badInstantReward);
    }

    /*
            when our robot is killed by the enemy
             */
    @Override
    public void onDeath(DeathEvent event) {
        backStep(badTerminalReward);
    }

    /*
        when our fired bullet hit the enemy
         */
    @Override
    public void onBulletHit(BulletHitEvent event) {
        backStep(goodInstantReward);
        numBulletHit++;
    }

    /*
        when the other robot is killed
         */
    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        backStep(goodTerminalReward);
    }

    private void backStep(double reward) {
        // we only perform back-step in IDLE mode, otherwise we do nothing (we expect this probability to be minimum)
        if(mode == RobotMode.IDLE) {
            numBackSteps++;

            // update Q value
            previousQ = myLUT.outputFor(previousStateAction);
            errorQ = ALPHA * (reward + GAMMA * currentQ - previousQ);
            if (learning) myLUT.train(previousStateAction, previousQ + errorQ);

            averageErrorQ += errorQ;
            // set the current state action pair as the previous one
            for(int i=0; i< NUM_STATES+NUM_ACTIONS; i++) {
                previousStateAction[i] = currentStateAction[i];
            }
        } else System.out.println("Oops! An event happened during SCAN mode");
        //change the state machine to SCAN mode again
        mode = RobotMode.SCAN;
    }

    /*
    The following classes handles battle over events and saves the LUT and statistics
     */

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        sampleCount++;
        /*
         auto-save for each 1000 samples
          */
//        if((sampleCount%AVERAGE_SAMPLE_SIZE == 0) && sampleCount!=0) {
//            try {
//                saveStats();    // save the statistics
//            } catch(IOException e){}
//        }

        // calculate average error Q
        averageErrorQ = averageErrorQ/numBackSteps;
        averageSumQ = averageSumQ/numBackSteps;

        try {
            saveStats();    // save the statistics
        } catch(IOException e){}

        // reset number of back step and averageErrorQ
        numBackSteps = 0;
        averageErrorQ = 0;
        averageSumQ = 0;
        accumulatedRewards = 0;

        numWallHit = 0;
        numBulletHit = 0;
        numHitByBullet = 0;

    }

    @Override
    public void onWin(WinEvent event) {
        numWins++;
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
//        try {
//            saveLUT();    // save the statistics
//        } catch(IOException e){}
    }

    private void saveStats() throws IOException {
        OutputStreamWriter w = new OutputStreamWriter(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        BufferedWriter writer = new BufferedWriter(w);
        writer.write(sampleCount + ", " + numWins + ", " + dfR.format(averageSumQ) + ", " + dfR.format(averageErrorQ) + ", "
                + numBackSteps + ", " + numWallHit + ", " + numHitByBullet + ", " + numBulletHit + "\n");
        writer.flush();
        writer.close();
    }

    private void saveLUT() throws IOException{
//        DataOutputStream writer = new DataOutputStream(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        OutputStreamWriter writer = new OutputStreamWriter(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        for(Map.Entry<List<Integer>, Double> entry: myLUT.getLookupTable().entrySet()) {
            for(Integer i : entry.getKey()) {
                writer.write(i);
            }
            writer.write('\t');
            writer.write(entry.getValue().toString());
            writer.write('\n');
            writer.flush();
        }
        writer.close();
    }
}
