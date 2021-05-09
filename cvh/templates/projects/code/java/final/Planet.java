/*
Programming Final Planet
ACO 101 -Intro to Computer Science
Christeprher Hunter
28 MAR 2021
*/

public class Planet {

    private static final double EARTHACCELERATION = 9.81;
    private String name;
    private double radius;
    private double mass;
    private double gravity;
    
    /**
     * CONSTRUCTOR w/ arguments
     * 
     * @param n Planet name
     * @param g Planet gravity
     * @param r Planet radius
     * @param m Planet mass
     */
    public Planet(String n, double g, double r, double m) {

        name = n;
        gravity = g;
        radius = r;
        mass = m;        
    }

    /**
     * Give the name entered upon construction
     * 
     * @return The name of the planet
     */
    public String getName() {

        return name;
    }

    /**
     * Give the gravity of the planet
     * 
     * @return Gravity on the surface of the planet
     */
    public double getGravity() {

        return gravity;
    }

    /**
     * Give the radius of the planet
     * 
     * @return Radius of the spherical planet
     */
    public double getRadius() {

        return radius;
    }

    /**
     * Give the given by construction
     * 
     * @return Mass of the planet
     */
    public double getMass() {

        return mass;
    }

    /**
     * Compute the volume of a planet
     * 
     * @return The volume of the planet object in km^3
     */
    public double volume() {

        return (4 / 3.0) * Math.PI * Math.pow(radius, 3);
    }

    /**
     * Compute the surface area of a planet
     * 
     * @return returns the surface area of the planet object in km^2
     */
    public double surfaceArea() {

        return 4 * Math.PI * Math.pow(radius, 2);
    }

    /**
     * Compute the acceleration of gravity on a planet
     * 
     * @return returns the average surface acceleration of the planet object
     */
    public double acceleration() {

        return  EARTHACCELERATION * getGravity();
    }

    /**
     * @param w weight of an object on Earth
     * @return returns the equivalent weight of a given weight on earth
     */
    public double weightOnPlanet(double w) {

        return w * getGravity();
    }

}
