

public class MotorBikeRunner {
  public static void main(String[] args) {
    MotorBike ducati = new MotorBike(100);
    MotorBike honda = new MotorBike(200);
    MotorBike somethingElse = new MotorBike();

    System.out.println(ducati.getSpeed());
    System.out.println(honda.getSpeed());
    System.out.println(somethingElse.getSpeed());

    ducati.start();
    honda.start();

    ducati.setSpeed(-100);
    System.out.println(ducati.getSpeed());
    // honda.setSpeed(80);
    // System.out.println(honda.getSpeed());
  }
  
}
class MotorBike {
  private int speed;

  MotorBike(){
    this(5);
  }

  MotorBike(int speed){
    this.speed = speed;
  }  
  public int getSpeed(){
    return speed;
  }
  public void setSpeed(int speed){
    if(speed > 0 )
      this.speed=speed;
  }

  public void increaseSpeed(int howMuch){
    setSpeed(this.speed +howMuch);
  }
  public void decreaseSpeed(int howMuch){
    
    setSpeed(this.speed-howMuch); 
    
  }
  
    void start(){
      System.out.println("Bike Start");
    }
}
