/*
Pop Quiz 01
Christerpher Hunter
28 JAN 2021
*/


public class PopQuiz_01 {
    
    public static void main(String[] args){

        String hypString = "zyphenated-word";

        int indexNum = hypString.indexOf("-");
        
        String str1 = hypString.substring(0, indexNum);
        String str2 = hypString.substring(indexNum +1);

        int compareVal = str1.compareTo(str2);

        System.out.printf("str1 is: %s%n", str1);
        System.out.printf("str2 is: %s%n", str2);

        if (compareVal > 0) {
            System.out.printf("%nstr1 is less then str2%n");
        } else if (compareVal == 0) {
            System.out.printf("%nstr1 is equivalent to str2%n");
        } else {
            System.out.printf("str1 is greater than str2%n");
        }


    }   
}