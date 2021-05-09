import java.util.Arrays;

/*
Programming Practice 14
Author: Christerpher Hunter   
Created: 2021MAR30
*/

public class ProgPrac14 {

    public static void main(String[] args) {

        int n = 5;

        int[][] squareArray = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    squareArray[i][j] = 1;
                }
            }
            System.out.printf("%s", Arrays.toString(squareArray[i]));
            System.out.println();
        }

    }
}
