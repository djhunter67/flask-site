/*
Programming Final Main
ACO 101 -Intro to Computer Science
Christeprher Hunter
27 MAR 2021
*/

import java.util.ArrayList;
import java.util.Scanner;

public class MainFile {
    
    public static void main(String[] args) {
        
        Menu mainSqueeze = new Menu();
        mainSqueeze.addListedOptions("Cash Register");
        mainSqueeze.addListedOptions("Magic 8-Ball");
        mainSqueeze.addListedOptions("Planets");
        mainSqueeze.addListedOptions("Exit");

        int choicePicked = mainSqueeze.getInput();
        switch (choicePicked){
            case 1:            
            registerTester();            
            break;
            case 2:          
            magic8BallTester();            
            break;
            case 3:            
            planetTester();            
            break;
            default:
            System.out.println("Exiting now");
            break;
        }                      
    }

    public static final void registerTester(){
        ArrayList<CRegister> register = new ArrayList<>();
        Scanner input = new Scanner(System.in);

        System.out.printf("%n%nHow many cash registers would you like to create: ");
        
        int numOfRegisters = input.nextInt();
        input.close();
        for (int i = 0; i < numOfRegisters; i++)
        {
            register.add(new CRegister());
            register.get(i).itemScannedCost(i * 1.25, i);
            register.get(i).amountPaid(i * 1.90); 
            System.out.printf("%n-----------------------------");
            System.out.printf("Register Number: %d%n", register.get(i).getRegisterNum());
            System.out.printf("Expected Register: %d%n", i);
            System.out.printf("%nTotals: $%,3.2f%n", register.get(i).totals());
            System.out.printf("Expected Totals: $%,3.2f%n", i * 1.25);
            System.out.printf("%nChange Due: $%,3.2f%n", register.get(i).changeDue());
            System.out.printf("Expected Change: $%,3.2f%n", i * (1.90 - 1.25));
            System.out.printf("%nRegister Serial: %5.0f%n%n", register.get(i).serialNumber());
        }                   
    }        

    public static final void magic8BallTester() {
        Scanner input = new Scanner(System.in);

        System.out.printf("%n%nPlease ask a question of the Magic 8 Ball: ");
        input.next();
        input.close();

        System.out.printf("%n%s%n%n", new Magic8Ball().ask());
    }

    public static final void planetTester(){
        
        Scanner input = new Scanner(System.in);

        String earthWeight = "";
        String whichPlanet = "";
        String idealOutput = "%n%-20s %,3.2f%n";        
        
        ArrayList<Planet> planets = new ArrayList<>();

        // Force in meters/second^2
        //
        final double[] gravity = { 0.39, 0.91, 1.00, 0.38, 2.87, 1.32, 0.93, 1.23 };

        // radius in Kilometers
        //
        final double[] radii = { 2439.7, 6051.8, 6378.1, 3396.2, 71492, 25559, 60268, 24764 };

        // Mass in 10^24 kg
        //
        final double[] mass = { 0.330, 4.87, 5.97, 0.642, 1898, 568, 86.8, 102 };

       
        do {
            // Fill and then empty on every iteration
            //
            for (int i = 0; i < gravity.length; i++){
                // Constructing each planet in the solar system        
                planets.add(new Planet(planetNames[i], gravity[i], radii[i], mass[i]));
            }

            System.out.printf("%nPlease enter a weight on Earth (0 to quit): ");
            earthWeight = input.next();

            if (earthWeight.startsWith("0")) {
                break;
            }

            double earthWeighted = Double.parseDouble(earthWeight);
            System.out.printf("%nPlease enter a planet name, Max, Min, or All: ");
            whichPlanet = input.next().toLowerCase();

            double[] planetVals = new double[planets.size()];
            for (int i = 0; i < planets.size(); i++) {
                planetVals[i] = planets.get(i).weightOnPlanet(earthWeighted);
            }
            switch (whichPlanet) {
            case "max":
                double maxVal = planetVals[0];
                for (int i = 0; i < planets.size(); i++) {
                    if (planetVals[i] > maxVal) {
                        maxVal = planetVals[i];
                    }
                }
                System.out.printf(idealOutput, planets.get(4).getName(), maxVal);
                break;
            case "min":
                double minVal = planetVals[0];
                for (int i = 0; i < planets.size(); i++) {
                    if (planetVals[i] < minVal) {
                        minVal = planetVals[i];
                    }
                }
                System.out.printf(idealOutput, planets.get(3).getName(), minVal);
                break;
            case "all":
                for (int i = 0; i < planets.size(); i++) {
                    System.out.printf(idealOutput,
                                      planets.get(i).getName(),
                                      planets.get(i).weightOnPlanet(earthWeighted));
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
                System.out.printf(idealOutput, 
                                  planets.get(findByPlanet(whichPlanet)).getName(),
                                  planets.get(findByPlanet(whichPlanet)).weightOnPlanet(earthWeighted));  
                System.out.printf(idealOutput, "Volume [km^3]",
                                  planets.get(findByPlanet(whichPlanet)).volume()); 
                System.out.printf(idealOutput, "Surface Area [km^2]",
                                  planets.get(findByPlanet(whichPlanet)).surfaceArea()); 
                System.out.printf(idealOutput, "Gravity [m/s^2]",
                                  planets.get(findByPlanet(whichPlanet)).acceleration());                          
                break;
            default:
                System.out.printf("%nINVALID INPUT%n");
                break;
            }
        planets.clear();
        } while (!earthWeight.startsWith("0"));
        input.close();    
    }

        // Planet names
        //
        static final String[] planetNames = {  "Mercury",
                                                "Venus", 
                                                "Earth", 
                                                "Mars", 
                                                "Jupiter", 
                                                "Saturn",
                                                "Uranus",
                                                "Neptune"
                                            };

        /**
         * Find the planet by their name
         * @param planet the name of the planet
         * @return the index of the array of that planet name
         */
        static int findByPlanet(String planetName){        
            for (int i = 0; i < planetNames.length; i++)
            {
                if (planetName.equalsIgnoreCase(planetNames[i]))
                {
                    return i;                
                } 
            }
            return -1;
        }
}
