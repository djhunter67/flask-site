/*
Programming Practice #3
Christerpher Hunter
21 JAN 2021
*/

import java.util.Scanner;
import javax.swing.*;

public class StringDemo {

    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {

        // Prompt user for a word
        //
        Scanner input = new Scanner(System.in);
        //System.out.print("Please enter at least a two letter word: ");
        String rand_word = JOptionPane.showInputDialog("Enter at least a two letter word:");
        input.close();

        // Display the word
        //
        //System.out.print("The word you have entered is : ");
        //System.out.printf(ANSI_RED + rand_word + ANSI_RESET + "%n%n");
        JOptionPane.showMessageDialog(null, "Word entered: " + rand_word);
        

        // Display the word with the first and last
        // character reversed; "radio" becomes "oadir"
        //
        int wrd_length = rand_word.length();
        String sub_string = rand_word.substring(1, wrd_length - 1);
        char fst_letter = rand_word.charAt(0);
        char lst_letter = rand_word.charAt(wrd_length - 1);
        //System.out.printf("%s%s%s", lst_letter, sub_string, fst_letter);
        JOptionPane.showMessageDialog(null, lst_letter + sub_string + fst_letter);
        
    }
    
}
