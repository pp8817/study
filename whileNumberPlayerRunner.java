public class whileNumberPlayerRunner {
  public static void main(String[] args) {
    WhileNumberPlayer player = new WhileNumberPlayer(30);

    player.printSquaresUptoLimit();
    //For limit = 30, output would be 1 4 9 16 25

    player.printCubesUptoLimit();
    //For limit = 30, output world be 1 8 27
  }
}


class WhileNumberPlayer{

  private int limit;

  public WhileNumberPlayer(int limit){
    this.limit = limit;
  }

  public void printSquaresUptoLimit(){
    //For limit = 30, output would be 1 4 9 16 25
    int i = 1;
    while(i*i<limit){
      System.out.print(i*i + " ");
      i++;
    }
    System.out.println();
  }

  public void printCubesUptoLimit(){
    //For limit = 30, output world be 1 8 27
    int i = 1;
    while(i*i*i<limit){
      System.out.print(i*i*i + " ");
      i++;
    }
    System.out.println();
  }
}