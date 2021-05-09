/*
Programming Practice #4
Christerpher Hunter
21 JAN 2021
*/

import java.util.Scanner;

public class ProgPrac4 {
    public static void main(String[] args) {
        
        Scanner input = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String str = input.next(); // or nextLine(), nextInt(), nextDouble()
        input.close();

        // Process user input

        char c = str.charAt(0);
        // get the length of the str
        int lengthOfStr = str.length();
        // get last character
        char lastChar = str.charAt(lengthOfStr - 1);
        // get substring of characters between first and last
        String sub_string = str.substring(1, lengthOfStr - 1);
        //concat str w/ the word "END"
        String str2 = str.concat("END");

        boolean hasItOrNaw = str.equals(str2);

        System.out.printf("%nCharacter at initial position: %s%n", c);
        System.out.printf("%nLength of str: %d%n", lengthOfStr);
        System.out.printf("%nCharacter at last position: %s%n", lastChar);
        System.out.printf("%nString between init and last characters: %s%n", sub_string);
        System.out.printf("%nAre str and str2 the same? %s%n", hasItOrNaw);

    }
}
