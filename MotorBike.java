public class MotorBike {
  private int speed;

  void setSpeed(int speed){
    this.speed = speed;
    System.out.println(speed);
    System.out.println(this.speed);
  }

  int getSpeed(){
    return this.speed;
  }
    void start(){
      System.out.println("Bike Start");
    }
}
