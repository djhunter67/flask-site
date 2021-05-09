/*
Programming Practrice 10
Author: Christerpher Hunter   
Created: 2021MAR09
*/

public class ProgPrac10 {
    public static void main(String[] args)
    {
        double length = 36.32;
        double width = 28.24;
        double height = 3.45;

        System.out.printf("%nThe box has the measurements %.3f Length, %.3f Width, and %.3f Height%n", length, width, height);
        System.out.printf("%nVolume: %.3f", boxVolume(length, width, height));
        System.out.printf("%nSurface Area: %.3f", boxSurfaceArea(length, width, height));
        System.out.printf("%nDiagonal Length: %.3f%n", boxDiagonal(length, width, height));
    }

    /**
     * Compute the Volume of a Cube
     * @param l length of the box
     * @param w width of the box
     * @param h height of the box
     * @return volume of the box
     */
    public static double boxVolume(double l, double w, double h)
    {
        return l * w * h;
    }
    
    /**
     * Compute the Surface Area of a box
     * @param l length of the box
     * @param w width of the box
     * @param h height of the box
     * @return surface area of the box
     */
    public static double boxSurfaceArea(double l, double w, double h)
    {
        return (2 * l * w) + (2 * l * h) + (2 * w * h);
    }

    /**
     * Compute the Diagonal Length of a box
     * @param l length of the box
     * @param w width of the box
     * @param h height of the box
     * @return diagonal length of a box
     */
    public static double boxDiagonal(double l, double w, double h)
    {
        double step1 = Math.pow(l, 2) + Math.pow(w, 2) + Math.pow(h, 2);

        return Math.sqrt(step1);
    }
}
