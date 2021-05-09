
import java.util.Scanner;

public class scan_prac
 {

    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
       Scanner usrInput = new Scanner(System.in);
       System.out.print("What is your name? ");
       
       String name = usrInput.nextLine();
       usrInput.close();

       System.out.println("Hello " + ANSI_RED + name + ANSI_RESET + " and you have succesfully run user input in Java!");
    }
}
