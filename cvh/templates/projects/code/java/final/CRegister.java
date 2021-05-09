/*
Programming Final Cash Register
ACO 101 -Intro to Computer Science
Christeprher Hunter
27 MAR 2021
*/

import java.util.ArrayList;

public class CRegister {

    private int registerNumber;
    private double totalDue;
    private double amountTendered;
    private double change;
    private ArrayList<Double> itemSerial;  
  
    /**
     * Constructs a brand new Cash Register
     */
    public CRegister() {
        registerNumber = 0;
        totalDue = 0.0;
        amountTendered = 0.0;
        change = 0.0;        
        itemSerial = new ArrayList<>();           
    }

    /**
     * Assigns a serial number to each cash register
     * @return the randomly assigned serial number
     */
    public double serialNumber(){
              
        double serial = Math.random()* 10E10;
        return serial;      
    }    

    /**
     * Take in the cost of the item and accrue the total cost
     * @param amount Cost of the item scanned
     * @param i Number of the cash register
     */
    public void itemScannedCost(double amount, int i){
        registerNumber += i;
        totalDue += amount;
        itemSerial.add(serialNumber());
    }

    /**
     * Take in the amount paid by the customer
     * @param monies Amount paid in doll hairs
     */
    public void amountPaid(double monies){

        amountTendered += monies;
    }

    /**
     * Keep track of the number of registers
     * @return The number of registers created
     */
    public int getRegisterNum(){       
            
            return registerNumber;
        }

   public double totals(){

        return totalDue;
    }

    public double changeDue(){

        change = amountTendered - totalDue;
        return change;
    }
          
}
