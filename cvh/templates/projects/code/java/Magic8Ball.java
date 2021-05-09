/*
Programming Practice 13
Author: Christerpher Hunter   
Created: 2021MAR25
*/

import java.util.Random;
import java.util.Scanner;

public class Magic8Ball {

    public static void main(String[] args) {

        Scanner input = new Scanner(System.in);

        System.out.print("Please ask a question of the Magic 8 Ball: ");
        input.close();
        
        System.out.printf("%n%s%n%n", ask());
    }

    /**
     * Take in user question and give magic 8 ball answer
     * @return index of the 'answer' array
     */
    public static String ask(){

        String[] answers = {"It is certain.", 
                            "It is decidely so.",
                            "Without a doubt.",
                            "Yes - definitely.",
                            "You may rely on it.",
                            "As I see it, yes.",
                            "Most likely.",
                            "Outlook good.",
                            "Yes.",
                            "Signs point to yes.",
                            "Reply hazy, try again.",
                            "Ask again later.",
                            "Better not tell you now.",
                            "Cannot predict now.",
                            "Concentrate and ask again.",
                            "Don't count on it.",
                            "My reply is no.",
                            "My sources say no.",
                            "Outlook not so good.",
                            "Very doubtful."
                            };
        
        Random rand = new Random();
        int randomNumber = rand.nextInt(19);

        return answers[randomNumber];
    }
    
}
