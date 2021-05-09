/*
Programming Final Menu
ACO 101 -Intro to Computer Science
Christeprher Hunter
27 MAR 2021
*/

import java.util.ArrayList;
import java.util.Scanner;

public class Menu {

    private ArrayList<String> choices;
    private Scanner input;

    /**
     * Constructs an empty menu
     */
    public Menu() {

        choices = new ArrayList<>();
        input = new Scanner(System.in);
    }

    /**
     * Take in items to add to the list
     * 
     * @param option Listed items
     */
    public void addListedOptions(String option) {

        choices.add(option);
    }

    /**
     * Displays the enumerated menu
     * 
     * @return The option, via integer, the user chose
     */
    public int getInput() {

        int userChoice;
        do {

            for (int i = 0; i < choices.size(); i++) {

                int pick = i + 1;
                System.out.printf("%n%d) %s%n", pick, choices.get(i));
            }
            System.out.println();

            userChoice = input.nextInt();

        } while (userChoice < 1 || userChoice > choices.size());        
        return userChoice;
    }
}
