/*
Programming Practice 12
Author: Christerpher Hunter   
Created: 2021MAR19
*/

import java.util.Scanner;

public class ProgPrac12 
{
    public static void main(String[] args) 
    {
        
        String userInput = "Please enter your salary";

        System.out.printf("%nThe amount entered: %5.2f", readDouble(userInput));
    }    

    /**
     * Take in a String and return a double
     * @param prompt String to be displayed
     * @return Double entered by user
     */
    public static double readDouble(String prompt)
    {
        Scanner input = new Scanner(System.in);

        System.out.printf("%n%s: ",prompt);
        double floatingInput = input.nextDouble();
        input.close();

        return floatingInput;
    }
}
