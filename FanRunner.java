public class FanRunner {
  public static void main(String[] args) {
    Fan fan = new Fan("Manufacturer 1", 0.34567,"GREEN");
    fan.switchOn();
    fan.setSpeed((byte)3);
    System.out.println(fan);
  }
}

class Fan{

  //state
  private String make;
  private double radius;
  private String color;
  private boolean isOn;
  private byte speed; // 1 to 5

  //creation
  public Fan(String make, double radius, String color){
    this.make = make;
    this.radius = radius;
    this.color = color;
  } 

  //isOn
  public void switchOn(){
    this.isOn = true;
    setSpeed((byte)5);
  }
  public void switchOff(){
    this.isOn = false;
    setSpeed((byte)0);
  }

  public void setSpeed(byte speed){
    this.speed = speed;
  }

  //print the state
  public String toString(){
    return String.format( "make - %s, radius - %f, color - %s, isOn - %b, speed - %d", make, radius, color, isOn, speed);
  }
}