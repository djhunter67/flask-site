/*
Programming Practice #5
Christerpher Hunter
28 JAN 2021
*/

import java.util.Scanner;

public class AgeDecades {

    public static void main(String[] args) {

        Scanner input = new Scanner(System.in);
        
        System.out.print("Please enter your age: ");
        int age = input.nextInt();
        input.close();

        if (age <=19) {
            System.out.printf("%nLess than Twenty%n");
        } else if ((age > 19) && (age <= 29)) {
            System.out.printf("%nIn your twenties%n");
        } else if (age > 29 && age <= 39) {
            System.out.printf("%nIn your thirties%n");
        } else if (age > 39 && age <= 49) {
            System.out.printf("%nIn your forties%n");
        } else if (age > 49 && age <= 59) {
            System.out.printf("%nIn your fifties%n");
        } else if (age > 59 && age <= 69) {
            System.out.printf("%nIn your sixties%n");
        } else {
            System.out.printf("%nIn your seventies or greater%n");
        } 
    
    }
}