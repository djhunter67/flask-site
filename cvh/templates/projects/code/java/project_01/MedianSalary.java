/*
Programming Project 01
ACO 101 -Intro to Computer Science
Christeprher Hunter
21 JAN 2021
*/

import java.util.Scanner;

public class MedianSalary {

    // Console color in order to highlight text
    //
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {

        // Take in the career and income of the user
        //
        Scanner input = new Scanner(System.in);

        String careerIncome;
        int maxIncome = 1000;
        int minIncome = 99999;
        String maxCareer = "";
        String minCareer = "";

        do {
            System.out.print("Please enter(title $salary): ");
            careerIncome = input.nextLine();

            if (careerIncome.isEmpty()) {
                System.out.printf("%nMaximum Salary: $%d for title: %s", maxIncome, maxCareer);
                System.out.printf("%nMinimum Salary: $%d for title: %s%n", minIncome, minCareer);
                break;

                // Catch no dollar sign symbol entered
                //
            } else if (!careerIncome.contains("$")) {
                System.out.printf("Invalid input - no $: %s%n", careerIncome);

                // Catch if no commas were entered
                //
            } else if (!careerIncome.contains(",")) {
                System.out.printf("Invalid input - no comma issues: %s%n", careerIncome);

            } else {

                final int income = careerIncome.indexOf("$");
                final int comma = careerIncome.indexOf(",");
                boolean notADigit = true;

                // Grab the number string entered by the user
                //
                String incomeString = careerIncome.substring(income + 1, comma) + careerIncome.substring(comma + 1);

                // Saving the career string only...
                //
                String career = "";
                if (careerIncome.charAt(0) == '$') {
                    System.out.printf("Invalid input - no title: %s%n", careerIncome);
                    notADigit = false;                    
                } else {
                    career = careerIncome.substring(0, income - 1).strip();
                }

                int commaCount = 0;
                boolean salaryNotNumeric = false;

                // Catch if too many commas were entered
                //
                for (int x = 0, j = careerIncome.substring(income).length(); x < j; x++) {
                    if (careerIncome.substring(income).charAt(x) == ',') {                        
                        commaCount += 1;                       
                    } 
                }

                // Check for 'not digits'
                //
                for (int i = 0, q = incomeString.length(); i < q; i++) {
                    if (Character.isDigit(incomeString.charAt(i))) {
                        notADigit = true;                        
                    } else {
                        salaryNotNumeric = true;
                        notADigit = false;
                    }
                }

                if (commaCount > 1) {
                    System.out.printf("Invalid input - comma issue: %s%n", careerIncome);
                    notADigit = false;
                } else if (salaryNotNumeric) {
                    System.out.printf("Invalid input - SalaryNotNumeric: %s%n", careerIncome);
                    notADigit = false;
                }

                // Append highest and lowest income
                //
                if (commaCount == 1 && notADigit) {

                    // Turn the income String into integers
                    //
                    int actualIncome = Integer.parseInt(incomeString);

                    // Retain the highest and lowest income entered
                    //
                    if (actualIncome > maxIncome) {
                        maxIncome = actualIncome;
                        maxCareer = career;
                    } else if (actualIncome < minIncome) {
                        minIncome = actualIncome;
                        minCareer = career;
                    }
                }
            }

        } while (!(careerIncome.isEmpty()));
        input.close();
    }

}
