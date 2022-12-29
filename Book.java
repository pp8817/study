public class Book {
  private int noOfCopies;
  Book(){}
  public Book(int noOfCopies){
    this.noOfCopies = noOfCopies;
  }

  public void setNoOfCopies(int noOfCopies){
    if(noOfCopies>0)
      this.noOfCopies = noOfCopies;
  }

  public void increaseSpeedNoOfCopies(int howMuch){
    setNoOfCopies(this.noOfCopies +howMuch);
  }
  public void decreaseSpeedNoOfCopies(int howMuch){
    setNoOfCopies(this.noOfCopies -howMuch);
  }
}
