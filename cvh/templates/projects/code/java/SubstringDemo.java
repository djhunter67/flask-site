/*
Determine if user entry contains a '/' character. Print the left substring and right substring if it does contain a '/', and print "No /" if it does not.
*/

import java.util.Scanner;

public class SubstringDemo
{
   public static void main(String[] args)
   {
      final char SLASH = '/';
      
      System.out.print("Enter a word or phrase: ");
      Scanner input = new Scanner(System.in);
      String entry = input.nextLine();
      input.close();
     
      final String SLASHED = Character.toString(SLASH);

      // finish the code
	if (entry.contains(SLASHED)) {
		int slash_ = entry.indexOf(SLASH);
		String leftString = entry.substring(0, slash_);
		String rightString = entry.substring(slash_ + 1);
		System.out.printf("%n%s%s%n", leftString, rightString);
	} else {
		// Not sure if the primitive type char SLASH would be in scope
		// SLASH is declared globally but I have run into issues here
		System.out.printf("%nNo %s%n", SLASH);
	    }


    }
}
