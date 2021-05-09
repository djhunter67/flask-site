/*
Programming Practice #5
Christerpher Hunter
28 JAN 2021
*/


public class ProgPrac5 {
    
    public static void main(String[] args){


        // There can only be WHOLE birds
        //
        int chicks = 0;
        int eggs = 0;
        int hens = 1;

        // Algorithm
        //
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
        eggs = hens;
        
        hens = chicks + hens;
        chicks = eggs;
                
        hens = chicks + hens;
             
                       
        System.out.printf("The final number of hens: %d%n", hens);
    }
}
