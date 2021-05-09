/*
Methods demonstration #1 - rectangle calculations: rectangleArea, rectanglePerimeter, rectangleDiagonal
Author: Christerpher Hunter   
Created: 2021MAR09
*/

public class MethodDemoV1Hunter {
   public static void main(String[] args) {
      double length = 3.0;
      double width = 7.0;

      double area = rectangleArea(length, width);
      double perimeter = rectanglePerimeter(length, width);
      double diagonal = rectangleDiagonal(length, width);

      System.out.printf("Area: %.1f   Perimeter: %.1f   Diagonal: %.1f", area, perimeter, diagonal);
   }

   /**
    * Compute the area of a rectangle
    * 
    * @param l the length of the rectangle
    * @param w the width of the rectangle
    * @return the area of the rectangle
    */
   public static double rectangleArea(double l, double w) {
      return l * w;
   }

   /**
    * Compute the perimeter of a rectangle
    * 
    * @param l the length of the rectangle
    * @param w the width of the rectangle
    * @return the perimeter of the rectangle
    */
   public static double rectanglePerimeter(double l, double w) {
      return l * 2 + w * 2;
   }

   /**
    * Compute the diagonal length of the rectangle
    * 
    * @param l the length of the rectangle
    * @param w the width of the rectangle
    * @return the hypotenuse of the triangle formed bby slicing the square
    *         diagonally
    */
   public static double rectangleDiagonal(double l, double w) {
      return Math.sqrt(Math.pow(l, 2) + Math.pow(w, 2));
   }
}