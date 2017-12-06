package bots;

import lookupTable.LUT;
import robocode.*;
import robocode.util.Utils;

import java.awt.*;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

import static robocode.util.Utils.normalRelativeAngle;

public class BasicWaveSurferBot extends AdvancedRobot{

    private static int BINS = 47;
    private static double _surfStats[] = new double[BINS];
    private Point2D.Double _myLocation;              // our bot's location
    private Point2D.Double _enemyLocation;           // enemy's location

    // Wave related
    private ArrayList _enemyWaves;
    private ArrayList _surfDirections;
    private ArrayList _surfAbsBearings;

    // Keep track of enemy energy
    private static double _oppEnergy = 100.0;


    /** This is a rectangle that represents an 800x600 battle field,
     * used for a simple, iterative WallSmoothing method (by PEZ).
     * If you're not familiar with WallSmoothing, the wall stick indicates
     * the amount of space we try to always have on either end of the tank
     * (extending straight out the front or back) before touching a wall.
     */
    private static Rectangle2D.Double _fieldRect
            = new java.awt.geom.Rectangle2D.Double(18, 18, 764, 564);
    private static double WALL_STICK = 140;
    // constants for arena dimension
    private static final double arenaWidth = 800;
    private static final double arenaHeight = 600;
    private static double enemyFired = 0;
    private static final double gunPower = 1.9;

    // Enum to index the action array
    private enum RobotActions {
        SURF, FIRE, CLOSE, AWAY
    }

    private RobotActions selectedAction;


    /*
    ===============================    LUT related variables    =============================
     state-action space
     5 States: enemy deltaX/Y from center, our robot deltaX/Y from center, enemy fired
     Quantization:
        enemy deltaX from center: -4~+4
        enemy deltaY from center: -3~+3
        robot delta X from center: -4~+4
        robot delta Y from center: -3~+3
        enemy fired: 0 - not fired (v<1); 1 - fired
     4 Actions: wave surfing; move away (from enemy); move closer to (enemy); fire
     State-Action space: 9*9*7*7*2*4 = 31932
      */

    private static double[][] stateActionTable = {
            {0, 0, 0, 0, 0, 1, 0, 0, 0},             // pre-populated for action surf
            {0, 0, 0, 0, 0, 0, 1, 0, 0},             // pre-populated for action fire
            {0, 0, 0, 0, 0, 0, 0, 1, 0},             // pre-populated for action move close
            {0, 0, 0, 0, 0, 0, 0, 0, 1},             // pre-populated for action move away
    };

    // Constants for LUT State actions
    private static final int NUM_STATES = 5;
    private static final int NUM_ACTIONS = 4;

    // initialize current and previous state action
    private static double[] currentStateAction = new double[NUM_STATES+NUM_ACTIONS];
    private static double[] previousStateAction = new double[NUM_STATES+NUM_ACTIONS];

    // x,y scaling factor (for quantization); for example, if scalingFactor = 0.01: x_states = 800 * 0.01 = 8;
    private static final double scalingFactor = 0.01;
    private static final double rewardFactor = 0.005;

    /*
Initialize the instance of look up table
 */
    private final static int[] floors = {
            (int) (-arenaWidth * scalingFactor / 2),
            (int) (-arenaHeight * scalingFactor / 2),
            (int) (-arenaWidth * scalingFactor / 2),
            (int) (-arenaHeight * scalingFactor / 2),
            0,
            0, 0, 0, 0   // lower bound for actions
    };

    private final static int[] ceilings = {
            (int) (+arenaWidth * scalingFactor / 2),
            (int) (+arenaHeight * scalingFactor / 2),
            (int) (+arenaWidth * scalingFactor / 2),
            (int) (+arenaHeight * scalingFactor / 2),
            0,
            1, 1, 1, 1   // upper bound for actions
    };

    // Discount factor and learning rate
    private static final double GAMMA = 0.9;
    private static final double ALPHA = 0.7;

    // previous and current Q value
    private static double currentQ = 0.0;

    // Control exploring and learning
    private static final boolean learning = true;
    private static final double epsilon = 0.05; // % exploration, >0 indicates exploration is turned on

    /*
    Statistics of learning
     */
    private static final int AVERAGE_SAMPLE_SIZE = 1000; // number of rounds which average is calculated
    private static int sampleCount = 0;
    private static int numWins = 0;
    // Total accumulated rewards
    private static double avgSumRewards = 0.0;
    private static int numBackSteps = 0;
    private static double averageSumQ = 0.0;
    // event callbacks
    private static int numWallHit = 0;
    private static int numBulletHit = 0;
    private static int numBulletHitBullet = 0;
    private static int numHitByBullet = 0;

    // initialize the instance of LUT
    private static LUT myLUT = new LUT(NUM_STATES + NUM_ACTIONS, floors, ceilings);

    /*
     Robocode Main
      */
    public void run() {
        _enemyWaves = new ArrayList();
        _surfDirections = new ArrayList();
        _surfAbsBearings = new ArrayList();

        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setBulletColor(Color.red);
        if (learning) setGunColor(Color.red);
        else setGunColor(Color.BLACK);

        // check the dimension of the arena, if it is not equal to the preset constants, raise an exception
        if (getBattleFieldHeight() != arenaHeight || getBattleFieldWidth() != arenaWidth) {
            throw new IllegalArgumentException("The actual battle field dimension is: " + getBattleFieldWidth() + " x " + getBattleFieldHeight());
        }

        // initialize the action to going up
        selectedAction = RobotActions.AWAY;

        do {
            turnRadarRightRadians(Double.POSITIVE_INFINITY);
        } while (true);
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        _myLocation = new Point2D.Double(getX(), getY());

        // our current lateral velocity
        double lateralVelocity = getVelocity()*Math.sin(e.getBearingRadians());
        double absBearing = e.getBearingRadians() + getHeadingRadians();

        // infinite lock
        setTurnRadarRightRadians(Utils.normalRelativeAngle(absBearing
                - getRadarHeadingRadians()) * 2);

        _surfDirections.add(0, (lateralVelocity >= 0) ? 1 : -1);
        _surfAbsBearings.add(0, absBearing + Math.PI);

        /*
         check if enemy has fired
          */
        double bulletPower = _oppEnergy - e.getEnergy();
        if (bulletPower < 3.01 && bulletPower > 0.09
                && _surfDirections.size() > 2) {
            // enemy has fired, we create a new wave
            EnemyWave ew = new EnemyWave();

            // we keep a record of fire time, velocity, distance traveled, direction, angle and the fire location
            ew.fireTime = getTime() - 1;
            ew.bulletVelocity = bulletVelocity(bulletPower);
            ew.distanceTraveled = bulletVelocity(bulletPower);
            ew.direction = (Integer) _surfDirections.get(2);
            ew.directAngle = (Double) _surfAbsBearings.get(2);
            ew.fireLocation = (Point2D.Double)_enemyLocation.clone(); // last tick

            _enemyWaves.add(ew);    // add the bullet to our wave collection
        }
        if(_enemyWaves.isEmpty()) enemyFired = 0;
        else enemyFired = 1;

        // update the enemy energy
        _oppEnergy = e.getEnergy();

        // update after EnemyWave detection, because that needs the previous
        // enemy location as the source of the wave
        _enemyLocation = project(_myLocation, absBearing, e.getDistance());

        updateWaves();

        // we update the state action until all actions are done
        if (getDistanceRemaining()==0 && getGunTurnRemaining() == 0 && getTurnRemaining() == 0){
            // update the state action table
            updateStateActionTable(e);

            // update the our current state action based on Q-value
            updateCurrentStateAction();

            // perform actions
            switch (selectedAction) {
                case FIRE:  // aim at the energy and fire
                    aimFire(e.getBearingRadians());
                    break;

                case SURF:    // do wave surfing
                    doSurfing();
                    break;

                case CLOSE:  // change heading to 180 (down)
                    move(false);
                    break;

                case AWAY:  // change heading to 270 (left)
                    move(true);
                    break;
            }
            execute();
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        // If the _enemyWaves collection is empty, we must have missed the
        // detection of this wave somehow.
        if (!_enemyWaves.isEmpty()) {
            Point2D.Double hitBulletLocation = new Point2D.Double(
                    e.getBullet().getX(), e.getBullet().getY());

            EnemyWave hitWave = null;
            // look through the EnemyWaves, and find one that could've hit us.
            for (Object _enemyWave : _enemyWaves) {
                EnemyWave ew = (EnemyWave) _enemyWave;

                if (Math.abs(ew.distanceTraveled -
                        _myLocation.distance(ew.fireLocation)) < 50
                        && Math.abs(bulletVelocity(e.getBullet().getPower())
                        - ew.bulletVelocity) < 0.001) {
                    hitWave = ew;
                    break;
                }
            }

            if (hitWave != null) {
                logHit(hitWave, hitBulletLocation);
                // We can remove this wave now, of course.
                _enemyWaves.remove(_enemyWaves.lastIndexOf(hitWave));
            }
        }

        /*
        learning related: calculate the reward and do a back-step
         */
        double bulletPower = e.getPower();
        double reward = -(Rules.getBulletDamage(bulletPower) + Rules.getBulletHitBonus(bulletPower))*rewardFactor;
        if(learning)backStep(reward);
        numBackSteps++;
        numHitByBullet++;
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        double reward = - 4 * rewardFactor;
        if(learning) backStep(reward);
        numBackSteps++;
        numWallHit++;
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        double reward = - 1 * rewardFactor;
        if(learning) backStep(reward);
        numBackSteps++;
    }

    @Override
    public void onDeath(DeathEvent event) {
        double reward = -100 * rewardFactor;
        if(learning) backStep(reward);
        numBackSteps++;
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        double reward = (Rules.getBulletHitBonus(gunPower)+Rules.getBulletDamage(gunPower))*rewardFactor;
        if(learning)backStep(reward);
        numBackSteps++;
        numBulletHit++;
    }

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        double reward = 100 * rewardFactor;
        if(learning) backStep(reward);
        numBackSteps++;
    }

    @Override
    public void onWin(WinEvent event) {
        numWins++;
    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        sampleCount++;
        // calculate average error Q
        averageSumQ = averageSumQ / numBackSteps;
        /*
         auto-save for each 500 samples
          */
        if ((sampleCount % AVERAGE_SAMPLE_SIZE == 0) && sampleCount != 0) {
            try {
                saveStats();    // save the statistics
            } catch (IOException e) {
                e.printStackTrace();
            }
            numWins = 0;
        }

        // reset number of back step and averageErrorQ
        avgSumRewards = 0;
        numBackSteps = 0;
        averageSumQ = 0;
        numBulletHitBullet = 0;
        numWallHit = 0;
        numBulletHit = 0;
        numHitByBullet = 0;
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        try {
            saveLUT();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*
                Learning related helper methods
                 */
    private void updateStateActionTable(ScannedRobotEvent event) {
        // update position variables
        double xFromCenter = getX() - arenaWidth / 2.0;
        double yFromCenter = getY() - arenaHeight / 2.0;
        double enemyXFromCenter = xFromCenter + Math.sin(event.getBearingRadians() + getHeadingRadians()) * event.getDistance();
        double enemyYFromCenter = yFromCenter + Math.cos(event.getBearingRadians() + getHeadingRadians()) * event.getDistance();

        // update the state action table
        for (RobotActions i : RobotActions.values()) {
            stateActionTable[i.ordinal()][0] = (double) Math.round(enemyXFromCenter * scalingFactor);
            stateActionTable[i.ordinal()][1] = (double) Math.round(enemyYFromCenter * scalingFactor);
            stateActionTable[i.ordinal()][2] = (double) Math.round(xFromCenter * scalingFactor);
            stateActionTable[i.ordinal()][3] = (double) Math.round(yFromCenter * scalingFactor);
            stateActionTable[i.ordinal()][4] = enemyFired;
        }
    }

    private void updateCurrentStateAction() {
        // store previous state action
        System.arraycopy(currentStateAction, 0, previousStateAction, 0, NUM_STATES + NUM_ACTIONS);

        // initialize max Q to -infinity
        double maxQ = Double.NEGATIVE_INFINITY;
        double tempQ;
        Random rand = new Random();

        // Get Q value for all (state, action) pairs for current state
        for (RobotActions action : RobotActions.values()) {
            // Use the LUT to find the Q values
            // select the action with max Q
            switch (action) {
                case SURF:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if (tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action : selectedAction;
                    break;

                case FIRE:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if (tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action : selectedAction;
                    break;

                case CLOSE:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if (tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action : selectedAction;
                    break;

                case AWAY:
                    tempQ = myLUT.outputFor(stateActionTable[action.ordinal()]);
                    if (tempQ > maxQ) {
                        maxQ = tempQ;
                        selectedAction = action;
                    } else if (tempQ == maxQ) selectedAction = (rand.nextBoolean()) ? action : selectedAction;
                    break;
            }
        }

        // The following will override the selected action if exploring is turned on
        if (epsilon > 0) {
            // epsilon greedy policy
            if (rand.nextDouble() < epsilon) {
                int i = rand.nextInt(NUM_ACTIONS);  //generates a random number between 0 and NUM_ACTIONS-1
                currentQ = maxQ;
                System.arraycopy(stateActionTable[i], 0, currentStateAction, 0, NUM_STATES + NUM_ACTIONS);
                selectedAction = RobotActions.values()[i];
            } else{
                currentQ = maxQ;
                System.arraycopy(stateActionTable[selectedAction.ordinal()], 0, currentStateAction, 0, NUM_STATES + NUM_ACTIONS);
            }
        } else { // go with the best Q
            currentQ = maxQ;
            System.arraycopy(stateActionTable[selectedAction.ordinal()], 0, currentStateAction, 0, NUM_STATES + NUM_ACTIONS);
        }
    }

    private void backStep(double reward) {
        numBackSteps++;
        // update Q value
        double previousQ = myLUT.outputFor(previousStateAction);
        double errorQ = ALPHA * (reward + GAMMA * currentQ - previousQ);
        myLUT.train(previousStateAction, previousQ + errorQ);

        // statistics
        averageSumQ += errorQ;
        avgSumRewards += reward;
    }

    /*
    Strategies helper
     */
    class EnemyWave {
        Point2D.Double fireLocation;
        long fireTime;
        double bulletVelocity, directAngle, distanceTraveled;
        int direction;

        EnemyWave() { }
    }

    /**
     * fast wall smoothing algorithm
     * x/y = current coordinates
     * startAngle = absolute angle that tank starts off moving - this is the angle
     *   they will be moving at if there is no wall smoothing taking place.
     * orientation = 1 if orbiting enemy clockwise, -1 if orbiting counter-clockwise
     * smoothTowardEnemy = 1 if smooth towards enemy, -1 if smooth away, 0 don't care
     */
    private double wallSmoothing(double x, double y, double startAngle, int orientation) {

        double angle = startAngle;

        // in Java, (-3 MOD 4) is not 1, so make sure we have some excess
        // positivity here
        angle += (4*Math.PI);

        double testX = x + (Math.sin(angle)*WALL_STICK);
        double testY = y + (Math.cos(angle)*WALL_STICK);
        double wallDistanceX = Math.min(x - 18, arenaWidth - x - 18);
        double wallDistanceY = Math.min(y - 18, arenaHeight - y - 18);
        double testDistanceX = Math.min(testX - 18, arenaWidth - testX - 18);
        double testDistanceY = Math.min(testY - 18, arenaHeight - testY - 18);

        double adjacent = 0;
        int g = 0; // because I'm paranoid about potential infinite loops

        while (!_fieldRect.contains(testX, testY) && g++ < 25) {
            if (testDistanceY < 0 && testDistanceY < testDistanceX) {
                // wall smooth North or South wall
                angle = ((int)((angle + (Math.PI/2)) / Math.PI)) * Math.PI;
                adjacent = Math.abs(wallDistanceY);
            } else if (testDistanceX < 0 && testDistanceX <= testDistanceY) {
                // wall smooth East or West wall
                angle = (((int)(angle / Math.PI)) * Math.PI) + (Math.PI/2);
                adjacent = Math.abs(wallDistanceX);
            }

            // use your own equivalent of (1 / POSITIVE_INFINITY) instead of 0.005
            // if you want to stay closer to the wall ;)
            angle += 1 *orientation*
                    (Math.abs(Math.acos(adjacent/WALL_STICK)) + 0.005);

            testX = x + (Math.sin(angle)*WALL_STICK);
            testY = y + (Math.cos(angle)*WALL_STICK);
            testDistanceX = Math.min(testX - 18, arenaWidth - testX - 18);
            testDistanceY = Math.min(testY - 18, arenaHeight - testY - 18);
        }

        return angle; // you may want to normalize this
    }

    private void aimFire(double enemyBearing) {
        double gunTurnAmount = normalRelativeAngle(enemyBearing + getHeadingRadians() - getGunHeadingRadians());
        setTurnGunRightRadians(gunTurnAmount);
        if (getGunHeat() == 0 && Math.abs(getGunTurnRemainingRadians()) < Math.toRadians(10))
            setFire(gunPower);
    }

    private static Point2D.Double project(Point2D.Double sourceLocation,
                                          double angle, double length) {
        return new Point2D.Double(sourceLocation.x + Math.sin(angle) * length,
                sourceLocation.y + Math.cos(angle) * length);
    }

    private static double absoluteBearing(Point2D.Double source, Point2D.Double target) {
        return Math.atan2(target.x - source.x, target.y - source.y);
    }

    private static double limit(double min, double value, double max) {
        return Math.max(min, Math.min(value, max));
    }

    private static double bulletVelocity(double power) {
        return (20.0 - (3.0*power));
    }

    private static double maxEscapeAngle(double velocity) {
        return Math.asin(8.0/velocity);
    }

    private static void setBackAsFront(AdvancedRobot robot, double goAngle) {
        double angle = Utils.normalRelativeAngle(goAngle - robot.getHeadingRadians());
        if (Math.abs(angle) > (Math.PI/2)) {
            if (angle < 0) {
                robot.setTurnRightRadians(Math.PI + angle);
            } else {
                robot.setTurnLeftRadians(Math.PI - angle);
            }
            robot.setBack(100);
        } else {
            if (angle < 0) {
                robot.setTurnLeftRadians(-1*angle);
            } else {
                robot.setTurnRightRadians(angle);
            }
            robot.setAhead(100);
        }
    }

    // update wave information
    private void updateWaves() {
        for (int x = 0; x < _enemyWaves.size(); x++) {      // for all enemy bullets
            EnemyWave ew = (EnemyWave)_enemyWaves.get(x);

            // update the distance traveled
            ew.distanceTraveled = (getTime() - ew.fireTime) * ew.bulletVelocity;

            // remove the bullet if it has traveled enough distance
            if (ew.distanceTraveled > _myLocation.distance(ew.fireLocation) + 50) {
                _enemyWaves.remove(x);
                x--;
//
//                // since we dodged a bullet do a back step here
//                double reward = 3*rewardFactor;
//                backStep(reward);
            }
        }
    }

    private EnemyWave getClosestSurfableWave() {
        double closestDistance = 50000; //initialize to a big number
        EnemyWave surfWave = null;

        for (Object _enemyWave : _enemyWaves) {
            EnemyWave ew = (EnemyWave) _enemyWave;

            // update the distance between our robot and the wave
            double distance = _myLocation.distance(ew.fireLocation)
                    - ew.distanceTraveled;

            // only surf if the bullet is not too close
            if (distance > ew.bulletVelocity && distance < closestDistance) {
                surfWave = ew;
                closestDistance = distance;
            }
        }

        return surfWave;
    }

    // Given the EnemyWave that the bullet was on, and the point where we
    // were hit, calculate the index into our stat array for that factor.
    private static int getFactorIndex(EnemyWave ew, Point2D.Double targetLocation) {
        double offsetAngle = (absoluteBearing(ew.fireLocation, targetLocation)
                - ew.directAngle);
        double factor = Utils.normalRelativeAngle(offsetAngle)
                / maxEscapeAngle(ew.bulletVelocity) * ew.direction;

        return (int)limit(0,
                (factor * ((BINS - 1) / 2)) + ((BINS - 1) / 2),
                BINS - 1);
    }

    // Given the EnemyWave that the bullet was on, and the point where we
    // were hit, update our stat array to reflect the danger in that area.
    private void logHit(EnemyWave ew, Point2D.Double targetLocation) {
        int index = getFactorIndex(ew, targetLocation);

        for (int x = 0; x < BINS; x++) {
            // for the spot bin that we were hit on, add 1;
            // for the bins next to it, add 1 / 2;
            // the next one, add 1 / 5; and so on...
            _surfStats[x] += 1.0 / (Math.pow(index - x, 2) + 1);
        }
    }

    private Point2D.Double predictPosition(EnemyWave surfWave, int direction) {
        Point2D.Double predictedPosition = (Point2D.Double)_myLocation.clone();
        double predictedVelocity = getVelocity();
        double predictedHeading = getHeadingRadians();
        double maxTurning, moveAngle, moveDir;

        int counter = 0; // number of ticks in the future
        boolean intercepted = false;

        do {    // the rest of these code comments are rozu's
            moveAngle = wallSmoothing(predictedPosition.x, predictedPosition.y, absoluteBearing(surfWave.fireLocation,
                            predictedPosition) + (direction * (Math.PI/2)), direction)
                            - predictedHeading;
            moveDir = 1;

            if(Math.cos(moveAngle) < 0) {
                moveAngle += Math.PI;
                moveDir = -1;
            }

            moveAngle = Utils.normalRelativeAngle(moveAngle);

            // maxTurning is built in like this, you can't turn more then this in one tick
            maxTurning = Math.PI/720d*(40d - 3d*Math.abs(predictedVelocity));
            predictedHeading = Utils.normalRelativeAngle(predictedHeading
                    + limit(-maxTurning, moveAngle, maxTurning));

            // this one is nice ;). if predictedVelocity and moveDir have
            // different signs you want to breack down
            // otherwise you want to accelerate (look at the factor "2")
            predictedVelocity +=
                    (predictedVelocity * moveDir < 0 ? 2*moveDir : moveDir);
            predictedVelocity = limit(-8, predictedVelocity, 8);

            // calculate the new predicted position
            predictedPosition = project(predictedPosition, predictedHeading,
                    predictedVelocity);

            counter++;

            if (predictedPosition.distance(surfWave.fireLocation) <
                    surfWave.distanceTraveled + (counter * surfWave.bulletVelocity)
                            + surfWave.bulletVelocity) {
                intercepted = true;
            }
        } while(!intercepted && counter < 500);

        return predictedPosition;
    }

    private double checkDanger(EnemyWave surfWave, int direction) {
        int index = getFactorIndex(surfWave,
                predictPosition(surfWave, direction));

        return _surfStats[index];
    }

    /*
    Robot actions
     */
    private void doSurfing() {
        EnemyWave surfWave = getClosestSurfableWave();

        if (surfWave == null) { return; }
        double dangerLeft = checkDanger(surfWave, -1);
        double dangerRight = checkDanger(surfWave, 1);

        double goAngle = absoluteBearing(surfWave.fireLocation, _myLocation);
        if (dangerLeft < dangerRight) {
            goAngle = wallSmoothing(_myLocation.x, _myLocation.y, goAngle - (Math.PI/2), -1);
        } else {
            goAngle = wallSmoothing(_myLocation.x, _myLocation.y, goAngle + (Math.PI/2), 1);
        }
        setBackAsFront(this, goAngle);
    }

    private void move(boolean away){
        // Calculate the go angle that we are expected to be still perpendicular to the enemy
        EnemyWave surfWave = getClosestSurfableWave();
        double goAngle = absoluteBearing(surfWave.fireLocation, _myLocation);

        // since we are moving away, we should calculate the new angle
        if (away) setBackAsFront(this, goAngle + Math.PI);
        else setBackAsFront(this, goAngle);
        _myLocation = new Point2D.Double(getX(), getY());
        updateWaves();
    }

    /*
    save and load
     */
    private void saveStats() throws IOException {
        DecimalFormat dfR = new DecimalFormat("0.000");
        OutputStreamWriter w = new OutputStreamWriter(new RobocodeFileOutputStream(getDataFile("statistics.txt").getAbsolutePath(), true));
        BufferedWriter writer = new BufferedWriter(w);
        writer.write(sampleCount + ", " + numWins + ", " + dfR.format(averageSumQ) + ", " + dfR.format(avgSumRewards) + ", "
                + numBackSteps + ", " + numWallHit + ", " + numHitByBullet + ", " + numBulletHit + ", " + numBulletHitBullet + "\n");
        writer.flush();
        writer.close();
    }

    private void saveLUT() throws IOException {
        DataOutputStream writerString = new DataOutputStream(new RobocodeFileOutputStream(getDataFile("LUT.txt").getAbsolutePath(), true));
        DataOutputStream writer = new DataOutputStream(new RobocodeFileOutputStream(getDataFile("LUT.dat").getAbsolutePath(), true));
        for (Map.Entry<String, Double> entry : myLUT.getLookupTable().entrySet()) {
                writerString.writeUTF(entry.getKey() + ", ");
                writerString.writeUTF(entry.getValue().toString() + "\n");

                writer.writeUTF(entry.getKey());
                writer.writeDouble(entry.getValue());
        }
        writerString.close();
        writer.close();
    }

    private void loadLUT() throws IOException {
        // Input stream to read in the lookup table values
        DataInputStream scanner = new DataInputStream(new FileInputStream(getDataFile("LUT.dat")));
        double[] X = new double[NUM_STATES+NUM_ACTIONS];
        double argValue;

        while(scanner.available()>0) {
            // read the lookup table values
            for (int i = 0; i < NUM_STATES+NUM_ACTIONS; i++) {
                X[i] = (double)scanner.readInt();
            }
            argValue = scanner.readDouble();
            myLUT.train(X,argValue);
        }
        scanner.close();
    }

}
