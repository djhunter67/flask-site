
public class ExamPrac {
    public static void main(String[] args) {

        int[] numbers = { 8, 18, 31, 7, 99 };

        StatsDemo objizzy = new StatsDemo();
        
        System.out.printf("%nAverage: %.2f", objizzy.average(numbers));

        System.out.printf("%nMIN: %d", objizzy.minimum(numbers));

        

        System.out.printf("%nMAX: %d", objizzy.maximum(numbers));

        System.out.printf("%nDiff Min: %d", minimum(12, 12));
    }

    private static int minimum(int val1, int val2){ 
        if (val1 > val2){ 
        return val2;
        } else if (val1 < val2){  
            return val1;   
        } else {return val1;}
    }

}


class StatsDemo {
    private int max;
    private int min;

    public StatsDemo() {
        max = 0;
        min = 0;
        
    }

    public double average(int[] array) {
        min = 0;
        for (int i = 0; i < array.length; i++) {
            min += array[i];
        }
        return min / (double)array.length;
    }

    // Not my best work here...
    public int minimum(int[] array) {
        
        for (int i = 0; i < array.length; i++) {
            min += array[i];
        }
        for (int j = 0; j < array.length; j++)
            if (array[j] < min) {
                min = array[j];
            }
        
        return min;
    }

    public int maximum(int[] array) {

        for (int i = 0; i < array.length; i++) {

            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
    }


}