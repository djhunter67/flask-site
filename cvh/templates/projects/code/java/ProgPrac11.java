/*
Programming Practice 11
Author: Christerpher Hunter   
Created: 2021MAR19
*/

public class ProgPrac11 {
    public static void main(String[] args) {
        int argument = 3456789;
        String testString = "This string has seven words, five average";

        System.out.printf("%nFirst Integer: %d", firstDigit(argument));
        System.out.printf("%nLast Integer: %d", lastDigit(argument));
        System.out.printf("%nNumber of digits: %d", digit(argument));
        System.out.printf("%nAverage of all letters: %d", countWords(testString));
        System.out.printf("%nMatches: %s", find("testing", "test"));
        System.out.printf("%nRecursive Number of Digits: %d", recursiveDigit(argument));

    }

    /**
     * find the first digit of the argument
     * 
     * @param n an integer value
     * @return the first digit of the argument
     */
    public static int firstDigit(int n) {
        String stringN = Integer.toString(n);

        char firstChar = stringN.charAt(0);
        int firstInt = firstChar - '0';

        return firstInt;
    }

    /**
     * find the last digit of the argument
     * 
     * @param n an integer value
     * @return the last digit of the argument
     */
    public static int lastDigit(int n) {
        String stringN = Integer.toString(n);

        char lastChar = stringN.charAt(stringN.length() - 1);
        int lastInt = lastChar - '0';

        return lastInt;
    }

    /**
     * find all the digits of the argument
     * 
     * @param n an integer value
     * @return the number of digits of the argument
     */
    public static int digit(int n) {
        String stringN = Integer.toString(n);
        int digits = stringN.length();

        return digits;
    }

    /**
     * Count the average length of the words in a sentence
     * 
     * @param string a sentence
     * @return the average length of all words
     */
    public static int countWords(String string) {
        int numberOfLetters = string.length();
        int totalBlanks = 0;
        for (int i = 0; i < string.stripTrailing().length(); i++) {
            if (string.charAt(i) == ' ') {
                totalBlanks++;
            }
        }
        return numberOfLetters / (totalBlanks + 1);
    }

    /**
     * Find a partial match recursively
     * 
     * @param string input string, one word
     * @param match  does it match? check here
     * @return Yay or Nay
     */
    public static boolean find(String string, String match) {
        if (string.length() < match.length())
            return false;

        if (string.substring(0, match.length()).equals(match))
            return true;

        return find(string.substring(1), match);

    }

    /**
     * find all the digits of the argument, recursively
     * 
     * @param n an integer value
     * @return the number of digits of the argument, recursivelt
     */
    public static int recursiveDigit(int n) {
        if (n < 10)
            return 1;
        else {
            return 1 + recursiveDigit(n / 10);
        }
    }
}