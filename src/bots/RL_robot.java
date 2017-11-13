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

import static robocode.util.Utils.normalRelativeAngle;
import static robocode.util.Utils.normalRelativeAngleDegrees;

public class RL_robot extends AdvancedRobot{

    private static final boolean debug = true;

    /*
    ===============================    LUT related variables    =============================
     */

    /*
     state-action space
     7 States: enemy velocity, enemy heading, enemy fired; distance from enemy, deltaX/Y from center, our robot heading;
     Quantization:
        velocity: 0 - stationary; 1 - moving
        enemy heading: quantized into 4 quadrants (0 - x<90, 1 - 90<=x<180, 2 - 180<=x<270, 3 - 270<=x)
        enemy fired: 0 - no; 1 - yes
        distance from enemy: 0~9
        delta X,Y from center: -4~4
        robot heading: quantized into 4 quadrants (0 - x<90, 1 - 90<=x<180, 2 - 180<=x<270, 3 - 270<=x)
     6 Actions: move up/down/left/right; fire (linear/circular)
      */

    private static double[][] stateActionTable = {
            {0, 0, 0, 0, 0, 0, 0, 0},             // pre-populated for action moveUp
            {0, 0, 0, 0, 0, 0, 0, 1},             // pre-populated for action moveDown
            {0, 0, 0, 0, 0, 0, 0, 2},             // pre-populated for action moveLeft
            {0, 0, 0, 0, 0, 0, 0, 3},             // pre-populated for action moveRight
            {0, 0, 0, 0, 0, 0, 0, 4},             // pre-populated for action fire (linear prediction)
//            {0, 0, 0, 0, 0, 0, 0, 5}              // pre-populated for action fire (circular prediction)
    };

    // Constants for LUT State actions
    private static final int NUM_STATES = 7;
    private static final int NUM_ACTIONS = 5;


    // Discount factor and learning rate
    private static final double GAMMA = 0.9;
    private static final double ALPHA = 0.7;

    // previous and current Q value
    private static double currentQ = 0.0;
    private static double previousQ = 0.0;
    private static double errorQ = 0.0;

    // Rewards weights
    private static final double badInstantReward = -0.25;
    private static final double badTerminalReward = -1;
    private static final double goodInstantReward = 0.25;
    private static final double goodTerminalReward = 1;

    // Control exploring and learning
    private static boolean learning = true;
    private static boolean exploring = false;
    private static boolean onPolicy = false;
    private static boolean enableInstantRewards = true;
    private static final double epsilon = 0.05; // % exploration when turned on

    /*
    Statistics of learning
     */
    private final int AVERAGE_SAMPLE_SIZE = 100; // number of rounds which average is calculated
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
    private static int numBulletDodge = 0;
    private static boolean isHitByBullet = false;
    private static double hitBulletPower = 0.0;
    private static boolean isBulletHit = false;
    private static boolean  removeDodgeCallback = false;

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

    // Enemy energy which is used to detect when the enemy has fired
    private static double currentEnemyEnergy;
    private static double previousEnemyEnergy = 100;

    // Last scan time
    private static long lastScanTime = 0;
    private static final long MAX_IDLE_TIME = 2;

    // Gun power used: 0.1 - 3
    private static final double gunPower = 1.9;

    // Distance moved each step
    private static final int stepDistance = 100;
    private static final int wallBuffer = 150;

    // x,y scaling factor (for quantization); for example, if scalingFactor = 0.01: x_states = 800 * 0.01 = 8;
    private static final double scalingFactor = 0.01;
    // constants for arena dimension
    private static final double arenaWidth = 800;
    private static final double arenaHeight = 600;

    // Enum to index the action array
    private enum RobotActions {UP, DOWN, LEFT, RIGHT, FIRE}
    private RobotActions selectedAction;

    // initialize current and previous state action
    private static double[] currentStateAction = new double[]{0, 0, 0, 0, 0, 0, 0, 0};
    private static double[] previousStateAction = new double[]{0, 0, 0, 0, 0, 0, 0, 0};


    // Enum of the state machine of the Robot, it first scan to get a sense of the current state s',
    // then select an action to maximize Q(s',a'), then it performs the action,
    // wait for the reward event when it updates the previous Q(s,a) and store s <- s', go back to scan mode
    private enum RobotMode {SCAN, SELECT, PERFORM, IDLE}
    private RobotMode mode;


    // store the enemy tank and our own tank status
    private static double ourRobotPositionX = 0.0;
    private static double ourRobotPositionY = 0.0;
    private static double currentEnemyPositionX = 0.0;
    private static double currentEnemyPositionY = 0.0;
    private static double currentEnemyHeading = 0.0;
    private static double currentEnemyVelocity = 0.0;
    private static double currentEnemyDistance = 0.0;
    private static double currentenemyBearingRadians = 0.0;
    private static double currentEnemyHeadingRadians = 0.0;
    private static boolean enemyFired = false;

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

    private LUT myLUT = new LUT(NUM_STATES+1, floors, ceilings);

    private Intercept intercept = new Intercept();


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
//            switch (mode){
//                case SCAN:
//                    turnRadarRight(45);
//                    break;
//
//                case SELECT:
//                    updateStateAction();
//                    mode = RobotMode.PERFORM;
//                    break;
//
//                case PERFORM:
//                    performAction();
//                    mode = RobotMode.SCAN;
//                    break;
//
//                case IDLE:
//                    // here we wait a bit for the rewards event to happen
//                    turnRadarRight(45);     // we still scan the enemy to make sure our radar still locks on the enemy
//                    break;
//            }

            // infinite lock
            if (getRadarTurnRemaining() == 0.0){
                setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
            }
            execute();

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
                System.arraycopy(stateActionTable[i], 0, currentStateAction, 0, NUM_STATES + 1);
                selectedAction = RobotActions.values()[i];
            }
        } else { // go with the best Q
            currentQ = maxQ;
            System.arraycopy(stateActionTable[selectedAction.ordinal()], 0, currentStateAction, 0, NUM_STATES + 1);
        }

        // add up the total Q
        averageSumQ += currentQ;
    }

    private void performAction(){
        double currentHeading = getHeading();

        // when the enemy has no energy left, we fire at it.
        if(currentEnemyEnergy == 0){
            setFire(gunPower);
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
                aimFire();
                break;

        }
        execute();
    }

    private void aimFire() {
        final double FIREPOWER = 2;
        final double ROBOT_WIDTH = 16,ROBOT_HEIGHT = 16;
        // Variables prefixed with e- refer to enemy, b- refer to bullet and r- refer to robot
        final double eAbsBearing = getHeadingRadians() + currentenemyBearingRadians;
        final double rX = getX(), rY = getY(),
                bV = Rules.getBulletSpeed(FIREPOWER);
        final double eX = rX + currentEnemyDistance*Math.sin(eAbsBearing),
                eY = rY + currentEnemyDistance*Math.cos(eAbsBearing),
                eV = currentEnemyVelocity,
                eHd = currentEnemyHeadingRadians;
        // These constants make calculating the quadratic coefficients below easier
        final double A = (eX - rX)/bV;
        final double B = eV/bV*Math.sin(eHd);
        final double C = (eY - rY)/bV;
        final double D = eV/bV*Math.cos(eHd);
        // Quadratic coefficients: a*(1/t)^2 + b*(1/t) + c = 0
        final double a = A*A + C*C;
        final double b = 2*(A*B + C*D);
        final double c = (B*B + D*D - 1);
        final double discrim = b*b - 4*a*c;
        if (discrim >= 0) {
            // Reciprocal of quadratic formula
            final double t1 = 2 * a / (-b - Math.sqrt(discrim));
            final double t2 = 2 * a / (-b + Math.sqrt(discrim));
            final double t = Math.min(t1, t2) >= 0 ? Math.min(t1, t2) : Math.max(t1, t2);
            // Assume enemy stops at walls
            final double endX = limit(
                    eX + eV * t * Math.sin(eHd),
                    ROBOT_WIDTH / 2, getBattleFieldWidth() - ROBOT_WIDTH / 2);
            final double endY = limit(
                    eY + eV * t * Math.cos(eHd),
                    ROBOT_HEIGHT / 2, getBattleFieldHeight() - ROBOT_HEIGHT / 2);
            setTurnGunRightRadians(robocode.util.Utils.normalRelativeAngle(
                    Math.atan2(endX - rX, endY - rY)
                            - getGunHeadingRadians()));
            setFire(FIREPOWER);
        }
    }

    private double limit(double value, double min, double max) {
        return Math.min(max, Math.max(min, value));
    }

    // we do predictive fire
    private void predictiveFire() {
        intercept.calculate(ourRobotPositionX, ourRobotPositionY, currentEnemyPositionX, currentEnemyPositionY, currentEnemyHeading,
                currentEnemyVelocity, gunPower, 0);
        double turnAngle = normalRelativeAngle(intercept.bulletHeading_deg - getGunHeading());
        turnGunRight(turnAngle);

        if (Math.abs(turnAngle) <= intercept.angleThreshold) {
            // Ensure that the gun is pointing at the correct angle
            if ((intercept.impactPoint.x > 0) && (intercept.impactPoint.x < getBattleFieldWidth()) &&
                    (intercept.impactPoint.y > 0) && (intercept.impactPoint.y < getBattleFieldHeight())) {
                // Ensure that the predicted impact point is within
                // the battlefield
                fire(gunPower);
            }
        }
    }

    private void updateStateActionTable(double enemyDistance) {
        // quantization
        double enemyVelocityQuantized = (currentEnemyVelocity != 0) ? 1 : 0;
        double enemyHeadingQuantized = degreeQuantization(currentEnemyHeading);
        double enemyFiredDouble = (enemyFired) ? 1 : 0;
        double enemyDistanceQuantized = (double)Math.round(enemyDistance * scalingFactor);
        double xFromCenterQuantized = (double)Math.round((getX() - arenaWidth/2.0) * scalingFactor);
        double yFromCenterQuantized = (double)Math.round((getY() - arenaHeight/2.0) * scalingFactor);
        double headingQuantized = degreeQuantization(getHeading());

        // update the state action table
        for(RobotActions i: RobotActions.values()){
            stateActionTable[i.ordinal()][0] = enemyVelocityQuantized;
            stateActionTable[i.ordinal()][1] = enemyHeadingQuantized;
            stateActionTable[i.ordinal()][2] = enemyFiredDouble;
            stateActionTable[i.ordinal()][3] = enemyDistanceQuantized;
            stateActionTable[i.ordinal()][4] = xFromCenterQuantized;
            stateActionTable[i.ordinal()][5] = yFromCenterQuantized;
            stateActionTable[i.ordinal()][6] = headingQuantized;
        }
    }

    private double degreeQuantization(double currentEnemyHeading) {
        if(currentEnemyHeading < 90){
            return 0;
        } else if (90 <= currentEnemyHeading && currentEnemyHeading < 180){
            return 1;
        } else if (180 <= currentEnemyHeading && currentEnemyHeading < 270){
            return 2;
        } else {
            return 3;
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

        if(x<wallBuffer){
            closeToLeftWall = true;
        } else if (x>(arenaWidth - wallBuffer)){
            closeToRightWall = true;
        }

        if(y<wallBuffer){
            closeToBottomWall = true;
        } else if (y>(arenaHeight - wallBuffer)){
            closeToTopWall = true;
        }
    }

    private void move(boolean closeToWall) {
        if(closeToWall){
            // TODO add wall avoidance
            setAhead(stepDistance);
        } else {
            setAhead(stepDistance);
        }
    }

    private void updateAllStats(ScannedRobotEvent enemyRobot) {
        ourRobotPositionX = getX();
        ourRobotPositionY = getY();
        currentEnemyDistance = enemyRobot.getDistance();
        currentenemyBearingRadians = enemyRobot.getBearingRadians();
        currentEnemyPositionX = ourRobotPositionX + Math.sin(enemyRobot.getBearingRadians() + getHeadingRadians()) * currentEnemyDistance;
        currentEnemyPositionY = ourRobotPositionY + Math.cos(enemyRobot.getBearingRadians() + getHeadingRadians()) * currentEnemyDistance;
        currentEnemyHeading = enemyRobot.getHeading();
        currentEnemyHeadingRadians = enemyRobot.getHeadingRadians();
        currentEnemyVelocity = enemyRobot.getVelocity();
        currentEnemyEnergy = enemyRobot.getEnergy();
    }

    /*
    callbacks: onScannedRobot, onHitByBullet, onBulletHit, onDeath, onWin
     */

    // This is the main callback where we update the states, select action and then perform action
    @Override
    public void onScannedRobot(ScannedRobotEvent enemyRobot) {
        // Turn Multiplier Lock
        double radarTurn = Utils.normalRelativeAngle(getHeadingRadians() + enemyRobot.getBearingRadians() - getRadarHeadingRadians());
        double extraTurn = Math.min(Math.atan(36.0/enemyRobot.getDistance()), Rules.RADAR_TURN_RATE_RADIANS);
        if(radarTurn < 0) radarTurn -= extraTurn;
        else radarTurn += extraTurn;
        setTurnRadarRightRadians(radarTurn);

        // update the stats of enemy robot and our robot
        updateAllStats(enemyRobot);
        double enemyDistance = enemyRobot.getDistance();

        // Check if the enemy has fired
        enemyFired = false;     // reset enemy fired

        // enemy energy drop
        double deltaEnemyEnergy = previousEnemyEnergy - currentEnemyEnergy;
        // we offset this energy drop when our bullet hits the enemy or we are hit by enemy bullet or the enemy runs into wall (ignored)
        if(isHitByBullet) {
            deltaEnemyEnergy += hitBulletPower;
            isHitByBullet = false;
        }
        if(isBulletHit) {
//            deltaEnemyEnergy -= (double)Math.round(Rules.getBulletDamage(gunPower));
            deltaEnemyEnergy -= 10.0;
            isBulletHit = false;
        }
        final double enemyGunpower = deltaEnemyEnergy;


        // we add a custom event listen to the event when we have dodged this bullet
        Condition dodgeBulletCallback = new Condition("dodgeBulletCallback") {
            @Override
            public boolean test() {
                double bulletSpeed = 20 - 3 * enemyGunpower;
                double bulletBearing = currentenemyBearingRadians;
                return ((bulletSpeed * (getTime() - enemyRobot.getTime()) > (enemyDistance+32 - getVelocity()*Math.sin(bulletBearing))) && !isHitByBullet && !removeDodgeCallback);
            }
        };
        if(deltaEnemyEnergy>1.9 && deltaEnemyEnergy<=3){ // when the enemy has fired a bullet
            mode = RobotMode.SCAN;
            enemyFired = true;
            isHitByBullet = false;
            addCustomEvent(dodgeBulletCallback);
            removeDodgeCallback = false;
        }
        if (removeDodgeCallback){   // when the flag to remove callback is set by event handler
            removeCustomEvent(dodgeBulletCallback);
        }

        previousEnemyEnergy = currentEnemyEnergy;
        // We only update the states and take actions when we are in scanning mode
        if(mode == RobotMode.SCAN) {
            // update scan time
            lastScanTime = enemyRobot.getTime();

            // update the current state action table
            updateStateActionTable(enemyDistance);

            // change the mode to SELECT action
            mode = RobotMode.SELECT;
            updateStateAction();

            // change the mode to perform actions
            mode = RobotMode.PERFORM;
            performAction();

            mode = RobotMode.IDLE;
        } else if (getTime() > lastScanTime + MAX_IDLE_TIME) mode = RobotMode.SCAN;
    }

    /*
    when we hit by a bullet, assign a negative reward and do a back step
     */
    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        isHitByBullet = true;
        backStep(-Rules.getBulletDamage(event.getPower()));
        hitBulletPower = Rules.getBulletHitBonus(event.getPower());
        numHitByBullet++;
    }

    /*
    when we hit the wall
     */
    @Override
    public void onHitWall(HitWallEvent event) {
        backStep(-Math.abs(getVelocity()*0.5 - 1));
        numWallHit++;
        // we step back from the wall
        double bearing = event.getBearing();
        if (bearing < 90 && bearing > -90){
            back(stepDistance);
        }else {
            ahead(stepDistance);
        }
    }

    /*
    when we hit the enemy
     */

    @Override
    public void onHitRobot(HitRobotEvent event) {
        backStep(-0.6);
    }

    /*
            when our robot is killed by the enemy
             */
    @Override
    public void onDeath(DeathEvent event) {
        backStep(-100);
    }

    @Override
    public void onBulletMissed(BulletMissedEvent event) {
        backStep(-gunPower);
    }

    /*
        when we dodged a bullet
         */
    @Override
    public void onCustomEvent(CustomEvent event) {
        removeDodgeCallback = true;
//        backStep(3);
//        numBulletDodge++;
    }

    /*
            when our fired bullet hit the enemy
             */
    @Override
    public void onBulletHit(BulletHitEvent event) {
        backStep(16);
        numBulletHit++;
        isBulletHit = true;
    }

    /*
        when the other robot is killed
         */
    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        backStep(100);
    }

    private void backStep(double reward) {
        // we only perform back-step in IDLE mode, otherwise we do nothing (we expect this probability to be minimum)
        numBackSteps++;

        // update Q value
        previousQ = myLUT.outputFor(previousStateAction);
        errorQ = ALPHA * (reward + GAMMA * currentQ - previousQ);
        if (learning) myLUT.train(previousStateAction, previousQ + errorQ);

        averageErrorQ += errorQ;
        // set the current state action pair as the previous one
        System.arraycopy(currentStateAction, 0, previousStateAction, 0, NUM_STATES + 1);

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
        if((sampleCount%AVERAGE_SAMPLE_SIZE == 0) && sampleCount!=0) {
//            try {
//                saveStats();    // save the statistics
//            } catch(IOException e){}
            numWins = 0;
        }

        // calculate average error Q
        averageErrorQ = averageErrorQ/numBackSteps;
        averageSumQ = averageSumQ/numBackSteps;

        // reset number of back step and averageErrorQ
        numBackSteps = 0;
        averageErrorQ = 0;
        averageSumQ = 0;
        accumulatedRewards = 0;

        numWallHit = 0;
        numBulletHit = 0;
        numHitByBullet = 0;
        numBulletDodge = 0;

    }

    @Override
    public void onWin(WinEvent event) {
        numWins++;
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        try {
            saveLUT();    // save the statistics
        } catch(IOException e){}
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
