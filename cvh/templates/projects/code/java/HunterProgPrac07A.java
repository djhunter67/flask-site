/*
Programming Practice #5
Christerpher Hunter
11 FEB 2021
*/

import java.util.Scanner;


public class HunterProgPrac07A {

    public static void main(String[] args) {

        Scanner input = new Scanner(System.in);

        String nameList = "";
        final String EXIT = "exit";
        String name = "";

        System.out.print("Please enter a name or enter \'exit\': ");
        String initName = input.nextLine();

        do {

            System.out.print("Please enter another name or enter \'exit\': ");
            name = input.nextLine();

            nameList += ", " + name;

            if (nameList.contains("exit")) {
                nameList = nameList.replace(", exit", "");
            }
                                   
        } while (!name.contains(EXIT));
        

        
        System.out.printf("The list of names: %s%s", initName.strip(), nameList.strip());

        ///////////////////////////////////////////////////////////////////////////////////

        final int LIMIT = 7;
        
        
        int loopCount = 1;
        long highVals = 1;

        System.out.printf("%n%nPlease enter an n to be a factorialized: ");
        int n = input.nextInt();
        
        System.out.printf("%nFactorial while loop%n");
        System.out.print("fn(0): 1");
        long startTime = System.nanoTime();
        do {        
           
            highVals = highVals * loopCount;
            loopCount++;
            System.out.printf("%nfn(%d): %d", loopCount - 1, highVals);

            if (loopCount == LIMIT + 1){
                break;
            }
                        
        } while (loopCount <= n);
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;

        System.out.printf("%n%nExecution time in seconds  %1.7f:",  timeElapsed / (Math.pow(10,9)));

        /////////////////////////////////////////////////////////////////////////////////

        
        long factorial = 1;
        System.out.printf("%n%nEnter an n to be factorialized: ");
        long userInput = input.nextLong();

        System.out.printf("%nFactorial for loop%n");
        System.out.print("fn(0): 1");
        long beginTime = System.nanoTime();
        for (long i = 1; i <= userInput; i++) {

            factorial = factorial * i;
            
            System.out.printf("%nfn(%d): %d", i, factorial);

            if (i == LIMIT) {
            break;
            }
        } 
        long stopTime = System.nanoTime();
        long timePassed = stopTime - beginTime;

        System.out.printf("%n%nExecution time in seconds  %1.7f:%n",  timePassed / (Math.pow(10,9)));


        input.close();     
    }   
}