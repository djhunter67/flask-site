import java.util.Scanner;

public class PopQuiz {
    public static void main(String[] args)
    {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter a number: ");
        double x = input.nextDouble();

        double y = Math.pow(x, 5) + Math.pow(x, 2) + 1;

        System.out.printf("%nOutput of the polynomial: %d.2f", y);

        input.close();
    }
}
