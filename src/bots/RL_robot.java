package bots;

import lookupTable.LUT;
import robocode.*;

import java.awt.Color;
import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

import static robocode.util.Utils.normalRelativeAngle;

public class RL_robot extends AdvancedRobot {

    private static final boolean debug = true;

    /*
    ===============================    LUT related variables    =============================
     */

    /*
     state-action space
     6 States: deltaX/Y from enemy, enemy velocity; deltaX/Y from center, energy level;
     Quantization:
        deltaX from enemy: -8~+8
        deltaY from enemy: -6~+6
        enemy velocity: 0 - stationary (v<1); 1 - moving
        delta X from center: -4~+4
        delta Y from center: -3~+3
        our own energy level: 0 - lower than 40; 1 - higher than 40
     5 Actions: move up/down/left/right; fire (linear/circular)
     State-Action space: 17*13*2*9*7*2*5 = 278460
      */

    private static double[][] stateActionTable = {
            {0, 0, 0, 0, 0, 0, 0},             // pre-populated for action moveUp
            {0, 0, 0, 0, 0, 0, 1},             // pre-populated for action moveDown
            {0, 0, 0, 0, 0, 0, 2},             // pre-populated for action moveLeft
            {0, 0, 0, 0, 0, 0, 3},             // pre-populated for action moveRight
            {0, 0, 0, 0, 0, 0, 4},             // pre-populated for action fire (linear prediction)
//            {0, 0, 0, 0, 0, 0, 0, 5}              // pre-populated for action fire (circular prediction)
    };

    // Constants for LUT State actions
    private static final int NUM_STATES = 6;
    private static final int NUM_ACTIONS = 5;


    // Discount factor and learning rate
    private static final double GAMMA = 0.9;
    private static final double ALPHA = 0.7;

    // previous and current Q value
    private static double currentQ = 0.0;

    // Control exploring and learning
    private static final boolean learning = true;
    private static final boolean exploring = true;
    private static final boolean onPolicy = false;
    private static final boolean terminalRewardOnly = false;
    private static final double epsilon = 0.25; // % exploration when turned on

    /*
    Statistics of learning
     */
    private static final int AVERAGE_SAMPLE_SIZE = 500; // number of rounds which average is calculated
    private static int sampleCount = 0;
    private static int numWins = 0;

    // Total accumulated rewards
    private static double accumulatedRewards = 0.0;
    private static double avgSumRewards = 0.0;

    private static int numBackSteps = 0;
    private static double averageSumQ = 0.0;
    private static double averageErrorQ = 0.0;

    // event callbacks
    private static int numWallHit = 0;
    private static int numBulletHit = 0;
    private static int numBulletHitBullet = 0;
    private static int numHitByBullet = 0;

    private static double totalNNSquaredError = 0;
    private static int winForAutosaving = 100; // automatically save the weights after number of wins

    /*
    Formatting log files
     */
    private DecimalFormat dfR = new DecimalFormat("0.000");


    /*
    ===============================    Robot control related variables    =============================
     */

    // Enemy energy which is used to detect when the enemy has fired
    private static double currentEnemyEnergy;
    private static double previousEnemyEnergy = 100;

    // Gun power used: 0.1 - 3
    private static final double gunPower = 2;

    // Distance moved each step
    private static final int stepDistance = 100;
    private static final int wallBuffer = 150;

    // x,y scaling factor (for quantization); for example, if scalingFactor = 0.01: x_states = 800 * 0.01 = 8;
    private static final double scalingFactor = 0.01;
    // constants for arena dimension
    private static final double arenaWidth = 800;
    private static final double arenaHeight = 600;
    private static double sumAvgErrorQ = 0;
    private static final double lowEnergyThreshold = 40.0;
    private static final boolean loadLUT = false;

    // Enum to index the action array
    private enum RobotActions {
        UP, DOWN, LEFT, RIGHT, FIRE
    }

    private RobotActions selectedAction;

    // initialize current and previous state action
    private static double[] currentStateAction = new double[NUM_STATES+1];
    private static double[] previousStateAction = new double[NUM_STATES+1];


    // Enum of the state machine of the Robot, it first scan to get a sense of the current state s',
    // then select an action to maximize Q(s',a'), then it performs the action,
    // wait for the reward event when it updates the previous Q(s,a) and store s <- s', go back to scan mode
    private enum RobotMode {
        SCAN, SELECT, PERFORM
    }

    private RobotMode mode;


    // store the enemy tank and our own tank status
    private static double currentEnemyVelocity = 0.0;
    private static double currentEnemyDistance = 0.0;
    private static double currentEnemyBearingRadians = 0.0;
    private static double currentEnemyHeadingRadians = 0.0;

    private boolean closeToTopWall = false, closeToLeftWall = false, closeToRightWall = false, closeToBottomWall = false;


    /*
    Initialize the instance of look up table
     */
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
//        myLUT.initialiseLUT();

        // we add a custom event listen to the event when the enemy has fired a bullet
//        Condition enemyFiredEvent = new Condition("enemyFiredEvent") {
//            @Override
//            public boolean test() {
//                return enemyFired;
////                double bulletSpeed = 20 - 3 * enemyGunpower;
////                double bulletBearing = currentEnemyBearingRadians;
////                return ((bulletSpeed * (getTime() - enemyRobot.getTime()) > (currentEnemyDistance+16 - getVelocity()*Math.sin(bulletBearing))) && !isHitByBullet && !removeDodgeCallback);
//            }
//        };
//        addCustomEvent(enemyFiredEvent);

        // Here we use Turn Multiplier Lock to ensure a lock on the enemy
//        turnRadarRightRadians(Double.POSITIVE_INFINITY);
        while(true) {
            switch (mode){
                case SCAN:
                    // wait until all actions have been performed and then we scan
                    if(getAllEvents().isEmpty()) {
                        setTurnRadarRight(45);
                        execute();
                    }
                    break;

                case SELECT:
                    // do a back-step
                    if(learning) backStep(accumulatedRewards);

                    // here we select the action based on state and maxQ (epsilon greedy)
                    updateStateAction();
                    mode = RobotMode.PERFORM;
                    break;

                case PERFORM:
                    performAction();
                    mode = RobotMode.SCAN;
                    break;
            }
        }
    }


    private void updateStateAction() {
        // store previous state action
        System.arraycopy(currentStateAction, 0, previousStateAction, 0, NUM_STATES + 1);

        // initialize max Q to -infinity
        double maxQ = Double.NEGATIVE_INFINITY;
        double tempQ;
        Random rand = new Random();

        // check if the robot is close to the wall: if it is, set the flags accordingly
        checkCloseToWall();

        // Get Q value for all (state, action) pairs for current state
        for (RobotActions action : RobotActions.values()) {
            // Use the LUT to find the Q values
            // select the action with max Q
            switch (action) {
                case FIRE:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if(tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action:selectedAction;
                    break;

                case LEFT:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if(tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    }else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action:selectedAction;
                    break;

                case RIGHT:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if(tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action:selectedAction;
                    break;

                case UP:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if(tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action:selectedAction;
                    break;

                case DOWN:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if(tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action:selectedAction;
                    break;
            }
        }

        // The following will override the selected action if exploring is turned on
        if (exploring) {
            // epsilon greedy policy
            if (rand.nextDouble() < epsilon) {
                int i = rand.nextInt(NUM_ACTIONS);  //generates a random number between 0 and NUM_ACTIONS-1
                if(onPolicy) currentQ = myLUT.outputFor(stateActionTable[i]);
                else currentQ = maxQ;
                System.arraycopy(stateActionTable[i], 0, currentStateAction, 0, NUM_STATES + 1);
                selectedAction = RobotActions.values()[i];
            } else{
                currentQ = maxQ;
                System.arraycopy(stateActionTable[selectedAction.ordinal()], 0, currentStateAction, 0, NUM_STATES + 1);
            }
        } else { // go with the best Q
            currentQ = maxQ;
            System.arraycopy(stateActionTable[selectedAction.ordinal()], 0, currentStateAction, 0, NUM_STATES + 1);
        }

        // add up the total Q
//        averageSumQ += currentQ;
    }

    private void performAction() {
        double currentHeading = getHeading();

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
        double gunTurnAmount = normalRelativeAngle(currentEnemyBearingRadians + getHeadingRadians() - getGunHeadingRadians());
        setTurnGunRightRadians(gunTurnAmount);
        if (getGunHeat() == 0 && Math.abs(getGunTurnRemainingRadians()) < Math.toRadians(10))
            setFire(gunPower);
    }

    private double limit(double value, double min, double max) {
        return Math.min(max, Math.max(min, value));
    }

    // we do predictive fire
    private void predictiveFire() {
        final double FIREPOWER = gunPower;
        final double ROBOT_WIDTH = 16, ROBOT_HEIGHT = 16;
        // Variables prefixed with e- refer to enemy, b- refer to bullet and r- refer to robot
        final double eAbsBearing = getHeadingRadians() + currentEnemyBearingRadians;
        final double rX = getX(), rY = getY(),
                bV = Rules.getBulletSpeed(FIREPOWER);
        final double eX = rX + currentEnemyDistance * Math.sin(eAbsBearing),
                eY = rY + currentEnemyDistance * Math.cos(eAbsBearing),
                eV = currentEnemyVelocity,
                eHd = currentEnemyHeadingRadians;
        // These constants make calculating the quadratic coefficients below easier
        final double A = (eX - rX) / bV;
        final double B = eV / bV * Math.sin(eHd);
        final double C = (eY - rY) / bV;
        final double D = eV / bV * Math.cos(eHd);
        // Quadratic coefficients: a*(1/t)^2 + b*(1/t) + c = 0
        final double a = A * A + C * C;
        final double b = 2 * (A * B + C * D);
        final double c = (B * B + D * D - 1);
        final double discrim = b * b - 4 * a * c;
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

    private void updateStateActionTable() {
        // quantization
        double xFromEnemy = (double) Math.round(Math.sin(currentEnemyBearingRadians + getHeadingRadians()) * currentEnemyDistance * scalingFactor);
        double yFromEnemy = (double) Math.round(Math.cos(currentEnemyBearingRadians + getHeadingRadians()) * currentEnemyDistance * scalingFactor);
        double enemyVelocity = (currentEnemyVelocity > 1) ? 1 : 0;
        double xFromCenterQuantized = (double) Math.round((getX() - arenaWidth / 2.0) * scalingFactor);
        double yFromCenterQuantized = (double) Math.round((getY() - arenaHeight / 2.0) * scalingFactor);
        double energyLevel = (lowEnergyThreshold < getEnergy()) ? 1 : 0;

        // update the state action table
        for (RobotActions i : RobotActions.values()) {
            stateActionTable[i.ordinal()][0] = xFromEnemy;
            stateActionTable[i.ordinal()][1] = yFromEnemy;
            stateActionTable[i.ordinal()][2] = enemyVelocity;
            stateActionTable[i.ordinal()][3] = xFromCenterQuantized;
            stateActionTable[i.ordinal()][4] = yFromCenterQuantized;
            stateActionTable[i.ordinal()][5] = energyLevel;
        }
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

        if (x < wallBuffer) {
            closeToLeftWall = true;
        } else if (x > (arenaWidth - wallBuffer)) {
            closeToRightWall = true;
        }

        if (y < wallBuffer) {
            closeToBottomWall = true;
        } else if (y > (arenaHeight - wallBuffer)) {
            closeToTopWall = true;
        }
    }

    private void move(boolean closeToWall) {
        if (closeToWall) {
            // TODO add wall avoidance
            setAhead(stepDistance);
        } else {
            setAhead(stepDistance);
        }
    }

    private void updateAllStats(ScannedRobotEvent enemyRobot) {
        currentEnemyDistance = enemyRobot.getDistance();
        currentEnemyBearingRadians = enemyRobot.getBearingRadians();
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
//        // Turn Multiplier Lock
//        double radarTurn = Utils.normalRelativeAngle(getHeadingRadians() + enemyRobot.getBearingRadians() - getRadarHeadingRadians());
//        double extraTurn = Math.min(Math.atan(36.0 / enemyRobot.getDistance()), Rules.RADAR_TURN_RATE_RADIANS);
//        if (radarTurn < 0) radarTurn -= extraTurn;
//        else radarTurn += extraTurn;
//        setTurnRadarRightRadians(radarTurn);

        // update the stats of enemy robot and our robot
        updateAllStats(enemyRobot);

        // Check if the enemy has fired
//        if (_bullets.isEmpty()) {
//            enemyFired = false;     // reset enemy fired
//        }
//        // enemy energy drop
//        double deltaEnemyEnergy = previousEnemyEnergy - currentEnemyEnergy;
//        // we offset this energy drop when our bullet hits the enemy or we are hit by enemy bullet or the enemy runs into wall (ignored)
//        if (isHitByBullet) {
//            deltaEnemyEnergy += hitBulletPower;
//            isHitByBullet = false;
//        }
//        if (isBulletHit) {
////            deltaEnemyEnergy -= (double)Math.round(Rules.getBulletDamage(gunPower));
//            deltaEnemyEnergy -= 10.0;
//            isBulletHit = false;
//        }
////        final double enemyGunpower = deltaEnemyEnergy;
//
//        if (deltaEnemyEnergy > 0.9 && deltaEnemyEnergy < 3.01) { // when the enemy has fired a bullet
//            // set the flag, trigger enemy fired event
//            enemyFired = true;
//            Point2D.Double bulletPosition = new Point2D.Double();
//            bulletPosition.setLocation(currentEnemyPositionX, currentEnemyPositionY);
//            Bullet bullet = new Bullet(enemyRobot.getTime(), currentEnemyBearingRadians + getHeadingRadians(), deltaEnemyEnergy, bulletPosition, currentEnemyDistance);
//            _bullets.add(bullet);
//
//            mode = RobotMode.SCAN;
//            isHitByBullet = false;
//        }
//
//        previousEnemyEnergy = currentEnemyEnergy;
        // We only update the current states when we are in scanning mode
        if (mode == RobotMode.SCAN) {
            // update the current state action table
            updateStateActionTable();

            // change the mode to SELECT action
            mode = RobotMode.SELECT;
        }
    }

    /*
    when we hit by a bullet, assign a negative reward and do a back step
     */
    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        double bulletPower = event.getPower();
//        hitBulletPower = Rules.getBulletHitBonus(bulletPower);
        double reward = -(Rules.getBulletDamage(bulletPower) + Rules.getBulletHitBonus(bulletPower));
        if(terminalRewardOnly) reward = 0;
//        if(learning) backStep(reward);
        accumulatedRewards+=reward;
        numHitByBullet++;

//        // we remove the bullet from the list
//        if (!_bullets.isEmpty()) _bullets.remove(0);
    }

    /*
    when we hit the wall
     */
    @Override
    public void onHitWall(HitWallEvent event) {
        double reward = -4;
        if(terminalRewardOnly) reward = 0;
        accumulatedRewards += reward;
//        if(learning) backStep(reward);

        numWallHit++;
    }

    /*
    when we hit the enemy
     */

    @Override
    public void onHitRobot(HitRobotEvent event) {
        double reward = -1;
        if(terminalRewardOnly) reward = 0;
        accumulatedRewards += reward;
//        if(learning) backStep(reward);
    }

    /*
            when our robot is killed by the enemy
             */
    @Override
    public void onDeath(DeathEvent event) {
        double reward = -100;
        if(learning) {
            double previousQ = myLUT.outputFor(currentStateAction);
            double errorQ = ALPHA * (GAMMA * reward - previousQ);
            myLUT.train(previousStateAction, previousQ + errorQ);
            averageErrorQ += errorQ;
            avgSumRewards += accumulatedRewards;
            accumulatedRewards = 0;
        }
    }

//    @Override
//    public void onBulletMissed(BulletMissedEvent event) {
//        double reward = -gunPower;
//        accumulatedRewards += reward;
////        if(learning)backStep(reward);
//    }

    /*
        when the enemy has fired a bullet
         */
//    @Override
//    public void onCustomEvent(CustomEvent event) {
//        if (debug) {
//            System.out.println("Enemy has fired a bullet!");
//        }
//
//        if(_bullets.isEmpty()) enemyFired = false;
//
//        // when we dodged the bullet, we remove it from the list
//        for (int i = 0; i < _bullets.size(); i++) {
//            Bullet bullet = (Bullet) _bullets.get(i);
//            if (bullet.getRemainingDistanceToRobot(event.getTime()) < 0) {
//                if (debug) {
//                    System.out.println("Bullet Dodged!");
//                }
//                _bullets.remove(i);
//
//                double reward = 3;
//                if(learning)backStep(reward);
//                accumulatedRewards += reward;
//                numBulletDodge++;
//            }
//        }
//        //change the state machine to SCAN mode again
//        mode = RobotMode.SCAN;
//    }

    /*
            when our fired bullet hit the enemy
             */
    @Override
    public void onBulletHit(BulletHitEvent event) {
        double reward = Rules.getBulletHitBonus(gunPower)+Rules.getBulletDamage(gunPower);
//        if(learning)backStep(reward);
        if(terminalRewardOnly) reward = 0;
        accumulatedRewards += reward;
        numBulletHit++;
    }

    @Override
    public void onBulletHitBullet(BulletHitBulletEvent event) {
        double reward = event.getHitBullet().getPower() - event.getBullet().getPower();
//        if(learning)backStep(reward);
        if(terminalRewardOnly) reward = 0;
        accumulatedRewards += reward;
        numBulletHitBullet++;
    }

    /*
            when the other robot is killed
             */
    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        double reward = 100;
        if(learning) {
            double previousQ = myLUT.outputFor(currentStateAction);
            double errorQ = ALPHA * (GAMMA * reward - previousQ);
            myLUT.train(previousStateAction, previousQ + errorQ);
            averageErrorQ += errorQ;
        }
        avgSumRewards += accumulatedRewards;
        accumulatedRewards = 0;
    }

    private void backStep(double reward) {
        numBackSteps++;

        // update Q value
        double previousQ = myLUT.outputFor(previousStateAction);
        double errorQ = ALPHA * (reward + GAMMA * currentQ - previousQ);
        myLUT.train(previousStateAction, previousQ + errorQ);

        averageErrorQ += errorQ;
        // set the current state action pair as the previous one
//        System.arraycopy(currentStateAction, 0, previousStateAction, 0, NUM_STATES + 1);

        avgSumRewards += accumulatedRewards;
        accumulatedRewards = 0;
    }

    /*
    The following classes handles battle over events and saves the LUT and statistics
     */

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        sampleCount++;
        /*
         auto-save for each 100 samples
          */
        if ((sampleCount % AVERAGE_SAMPLE_SIZE == 0) && sampleCount != 0) {
            sumAvgErrorQ = sumAvgErrorQ/AVERAGE_SAMPLE_SIZE;
            avgSumRewards = avgSumRewards/AVERAGE_SAMPLE_SIZE;
            try {
                saveStats();    // save the statistics
            } catch (IOException e) {
                e.printStackTrace();
            }
            numWins = 0;
            sumAvgErrorQ = 0;
            avgSumRewards = 0;
        }

        // calculate average error Q
        averageErrorQ = averageErrorQ / numBackSteps;
        sumAvgErrorQ += averageErrorQ;
        averageSumQ = averageSumQ / numBackSteps;

        // reset number of back step and averageErrorQ
        numBackSteps = 0;
        averageErrorQ = 0;
        averageSumQ = 0;
        accumulatedRewards = 0;
        numBulletHitBullet = 0;

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
        try {
            saveLUT();    // save the LUT
        } catch (IOException e) { e.printStackTrace(); }
    }

    private void saveStats() throws IOException {
        OutputStreamWriter w = new OutputStreamWriter(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        BufferedWriter writer = new BufferedWriter(w);
        writer.write(sampleCount + ", " + numWins + ", " + dfR.format(sumAvgErrorQ) + ", " + dfR.format(avgSumRewards) + ", "
                + numBackSteps + ", " + numWallHit + ", " + numHitByBullet + ", " + numBulletHit + ", " + numBulletHitBullet + "\n");
        writer.flush();
        writer.close();
    }

    private void saveLUT() throws IOException {
//        DataOutputStream writer = new DataOutputStream(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        boolean writeString = true;

        DataOutputStream writer = new DataOutputStream(new RobocodeFileOutputStream(getDataFile("LUT.txt").getAbsolutePath(), true));
        for (Map.Entry<String, Double> entry : myLUT.getLookupTable().entrySet()) {
//            for (int i : entry.getKey()) {
//                if(writeString) writer.writeUTF(String.valueOf(i) + ", ");
//                else writer.writeInt(i);
//            }
            if(writeString){
                writer.writeUTF(entry.getKey() + ", ");
                writer.writeUTF(entry.getValue().toString() + "\n");
            }
            else {
                writer.writeChars(entry.getKey());
                writer.writeDouble(entry.getValue());
            }
        }
        writer.close();
    }

    private void loadLUT() throws IOException {
        // Input stream to read in the lookup table values
        DataInputStream scanner = new DataInputStream(new FileInputStream(getDataFile("LUT.dat")));
        double[] X = new double[NUM_STATES+1];
        double argValue;

        while(scanner.available()>0) {
            // read the lookup table values
            for (int i = 0; i < NUM_STATES+1; i++) {
                X[i] = (double)scanner.readInt();
            }
            argValue = scanner.readDouble();
            myLUT.train(X,argValue);
        }
        scanner.close();
    }
}
