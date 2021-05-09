/*
Programming Final 8 Ball
ACO 101 -Intro to Computer Science
Christeprher Hunter
27 MAR 2021
*/

import java.util.ArrayList;
import java.util.Random;

public class Magic8Ball {

    private ArrayList<String> answers = new ArrayList<>();
    
    /**
     * Take in user question and give magic 8 ball answer
     * @return index of the 'answer' array
     */
    public String ask(){

        answers.add("It is certain.");
        answers.add("It is decidely so.");
        answers.add("Without a doubt."); 
        answers.add("Yes - definitely."); 
        answers.add("You may rely on it."); 
        answers.add("As I see it, yes."); 
        answers.add("Most likely."); 
        answers.add("Outlook good."); 
        answers.add("Yes."); 
        answers.add("Signs point to yes."); 
        answers.add("Reply hazy, try again."); 
        answers.add("Ask again later."); 
        answers.add("Better not tell you now."); 
        answers.add("Cannot predict now."); 
        answers.add("Concentrate and ask again."); 
        answers.add("Don't count on it."); 
        answers.add("My reply is no."); 
        answers.add("My sources say no."); 
        answers.add("Outlook not so good."); 
        answers.add("Very doubtful."); 
               
        Random rand = new Random();
        int randomNumber = rand.nextInt(19);

        return answers.get(randomNumber);
    
    }
    
    
    
}