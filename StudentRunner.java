import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;

public class StudentRunner {
  public static void main(String[] args) {
    Student student = new Student("Ranga",97, 98, 100);

    int number = student.getNumberOfMarks();
    int sum = student.getTotalSumOfMarks();
    int maximumMark = student.getMaximumMark();
    int minimumMark = student.getMinimumMark();
    System.out.println(number);
    System.out.println(sum);
    System.out.println(maximumMark);
    System.out.println(minimumMark);
    BigDecimal average = student.getAverageMarks();
    System.out.println(average);
    student.addNewMark(35);
    student.removeMarkAtIndex(1);
    System.out.println(student);
  }
}

class Student{
  private String name;
  private ArrayList<Integer> marks = new ArrayList<Integer>();

  public Student(String name, int... marks){
    this.name = name;
    for(int mark:marks){
      this.marks.add(mark);
    }
  }

  public int getNumberOfMarks(){
    return marks.size();
  }

  public int getTotalSumOfMarks(){
    int sum = 0;
    for(int mark:marks){
      sum += mark;
    }
    return sum;
  }
  public int getMaximumMark(){
    int max = Integer.MIN_VALUE;
    for (int mark:marks){
      if(mark>max){
        max = mark;
      }
    }
    return max;
  }
  public int getMinimumMark(){
    int mini = Integer.MAX_VALUE;
    for (int mark:marks){
      if(mark<mini){
        mini = mark;
      }
    }
    return mini;
  }

  public BigDecimal getAverageMarks(){
    int sum = getTotalSumOfMarks();
    int number = getNumberOfMarks();

    return new BigDecimal(sum/number);
  }

  public String toString(){
    return name + marks;
  }

  public void addNewMark(int mark){
    marks.add(mark);
  }

  public void removeMarkAtIndex(int index){
    marks.remove(index);
  }
}