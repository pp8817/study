public class MyCharRunner {
  public static void main(String[] args) {
    MyChar myChar =new MyChar('A');
    System.out.println(myChar.isVowel());
  }
}

class MyChar{
  private char ch;

  public MyChar(char ch) {
    this.ch = ch;
  }


  public boolean isVowel() {
    if(ch =='a' ||ch =='e' ||ch =='i' ||ch =='o' ||ch =='u')
      return true;
    if(ch =='A' ||ch =='E' ||ch =='I' ||ch =='O' ||ch =='U')
      return true;
    
    return false;
    
  }
}