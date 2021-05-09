/*
Programming Project 02
ACO 101 -Intro to Computer Science
Christeprher Hunter
24 FEB 2021
*/

import java.util.Scanner;

public class PlanetWeight {
        
    public static void main(String[] args) {

        Scanner input = new Scanner(System.in);
        String earthWeight = "";
        String whichPlanet = "";
        String idealOutput = "%n%-20s %3.2f%n";
        
        
        do  {                           
            System.out.printf("%nPlease enter a weight on Earth (0 to quit): ");
            earthWeight = input.next();
            
            if (earthWeight.startsWith("0")) 
            {
                break;
            }
        
            double earthWeighted = Double.parseDouble(earthWeight);     
            System.out.printf("%nPlease enter a planet name, Max, Min, or All: ");
            whichPlanet = input.next().toLowerCase();

            double[] planetVals = new double[planets.length];
            for (int i = 0; i < planets.length; i++)
                {                        
                    planetVals[i] = weightOnPlanet(earthWeighted, i);                        
                }
            switch (whichPlanet)
            {
            case "max":                    
                double maxVal = planetVals[0];
                for (int i = 0; i < planets.length; i++) {
                    if (planetVals[i] > maxVal) 
                    {
                        maxVal = planetVals[i];
                    }
                }                    
                System.out.printf(idealOutput, planets[4], maxVal);
            break;
            case "min":
                double minVal = planetVals[0];
                for (int i = 0; i < planets.length; i++) {
                    if (planetVals[i] < minVal) 
                    {
                        minVal = planetVals[i];
                    }
                }
                System.out.printf(idealOutput, planets[3], minVal);
            break;
            case "all":
                for (int i = 0; i < planets.length; i++)
                {
                    System.out.printf(idealOutput, planets[i], weightOnPlanet(earthWeighted, i));
                }
            break;                
            case "mercury":                 
            case "venus":                
            case "earth":               
            case "mars":               
            case "jupiter":                
            case "saturn":               
            case "uranus":                
            case "neptune":
                System.out.printf(idealOutput, planets[findByPlanet(whichPlanet)], weightOnPlanet(earthWeighted, findByPlanet(whichPlanet)));
            break;
            case "0.39":                          
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(0.39)));
            break;
            case "0.91":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(0.91)));
            break;
            case "1.00":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(1.00)));
            break;
            case "0.38":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(0.38)));
            break;
            case "2.87":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(2.87)));
            break;
            case "1.32":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(1.32)));
            break;
            case "0.93":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(0.93)));
            break;
            case "1.23":
                System.out.printf(idealOutput, planets[findByGravity(Double.parseDouble(whichPlanet))], weightOnPlanet(earthWeighted, findByGravity(1.23)));
            break;
            default: 
                System.out.printf("%nINVALID INPUT%n");
            break;                 
            }
                     
        } while (!earthWeight.startsWith("0"));
        input.close();
    }

    private static final String[] planets = { 
                                            "Mercury",
                                            "Venus",
                                            "Earth",
                                            "Mars",
                                            "Jupiter",
                                            "Saturn",
                                            "Uranus",
                                            "Neptune"};

    private static final double[] gravity = {
                                            0.39,
                                            0.91,
                                            1.00,
                                            0.38,
                                            2.87,
                                            1.32,
                                            0.93,
                                            1.23};    

    /**
     * Find the planet gravity by their name
     * @param planet the name of the planet
     * @return the planets gravity ratio
     */
    public static final int findByPlanet(String planet)
    {        
        for (int i = 0; i < planets.length; i++)
        {
            if (planet.equalsIgnoreCase(planets[i]))
            {
                return i;                
            } 
        }
        return -1;
    }       

    /**
     * Find the planet via gravity
     * @param grvty the gravity of the planet
     * @return something about the planey via gravity
     */
    public static final int findByGravity(double grvty)
    {        
        for (int i = 0; i < gravity.length; i++)
        {
            if (grvty == gravity[i])
            {
                return i;                
            } 
        }
        return -1;
    }

    /**
     * Find how much one would weigh on a given planet
     * @param weight how much an item weighs on Earth
     * @param planet which planet by number in order from the sun
     * @return a decimal value of weight in Kg
     */
    public static final double weightOnPlanet(double weight, int planet)
    {
        return weight * gravity[planet];
    }


}
