package bots;

import java.awt.geom.Point2D;

public class Bullet {
    private static long fireTime;
    private static double absBearing;
    private static double bulletPower;
    private static Point2D.Double firePosition;
    private static double distanceToRobot;

    Bullet(long fireTime, double absBearing, double bulletPower, Point2D.Double firePosition, double distanceToRobot){
        this.fireTime = fireTime;
        this.firePosition = firePosition;
        this.bulletPower = bulletPower;
        this.absBearing = absBearing;
    }

    double getBulletDistanceTraveled(long presentTime){
        return (presentTime - fireTime)*(20-3*bulletPower);
    }

    double getRemainingDistanceToRobot(long presentTime){
        return distanceToRobot - getBulletDistanceTraveled(presentTime);
    }

    Point2D.Double getBulletCurrentLocation(long distance){
        Point2D.Double currentPosition = new Point2D.Double();
        currentPosition.setLocation(firePosition.getX() - distance * Math.sin(absBearing), firePosition.getY() - distance * Math.cos(absBearing));
        return currentPosition;
    }
}
