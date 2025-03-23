// 자바 자료형 공부

public class MyCharRunner {
  public static void main(String[] args) {
    MyChar myChar = new MyChar('A');
    System.out.println(myChar.isVowel());
    // myChar.isConsonant();
    System.out.println(myChar.isDigit());
    System.out.println(myChar.isAlphabet());

    System.out.println(myChar.isConsonant());
    // MyChar.printLowerCaseAlphabets();
    // MyChar.printUpperCaseAlphabets();
  }
}

class MyChar{
  private char ch;

  public MyChar(char ch) {
    this.ch = ch;
  }


  public boolean isVowel() {
    if(ch =='a' || ch =='e' || ch =='i' || ch =='o' || ch =='u')
      return true;
    if(ch =='A' || ch =='E' || ch =='I' || ch =='O' || ch =='U')
      return true;
    
    return false;
    
  }

  public boolean isDigit(){
    if (ch >= 48 && ch <= 57) //betweeb '0 and '9'
     return true;
    return false;
  }
  public boolean isAlphabet(){
    if (ch >= 97 && ch <= 122) //betweeb 'a' and 'z'
    return true;
    if (ch >= 65 && ch <= 90) //betweeb 'A' and 'Z'
     return true;
    return false;
  }

  public boolean isConsonant(){
    //Alphabet and it is not Vowel
    //! [a, e, i, o, u]
    if(isAlphabet() && !isVowel())
      return true;
    return false;
  }

  public static void printLowerCaseAlphabets(){
  // a to z
    for (char ch = 'a'; ch <= 'z'; ch++){
      System.out.println(ch);
    }
  }
  public static void printUpperCaseAlphabets(){
    // a to z
    for (char ch = 'A'; ch <= 'Z'; ch++){
      System.out.println(ch);
    }
  }
}