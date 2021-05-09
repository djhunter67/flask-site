class NumsPopQuiz {
    public static void main(String[] args) {

        int[] nums = {2, 4, 1, 5, 6, 9, 4, 12, 1, 54, 35};

        int min = 99;
        int max = 0;
        int sum = 0;

        for (int i = 0; i < nums.length; i++) {

            if (nums[i] < min) {min = nums[i];}
            else if (nums[i] > max) {max = nums[i];}

            sum += nums[i];
        }
        System.out.printf("%nMAX: %d%nMIN: %d", max, min);
        System.out.printf("%n%d", sum / nums.length);
        
    }
}