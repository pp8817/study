import java.math.BigDecimal;


class Simple {

  BigDecimal principal;
  BigDecimal interest;

  public Simple(String principal, String interest){
    this.principal = new BigDecimal(principal);
    this.interest = new BigDecimal(interest).divide(new BigDecimal(100));
  }
  public BigDecimal calculatorTotalValue(int noOfYear) {
    BigDecimal noOfYearsBigDecimal;
    noOfYearsBigDecimal = new BigDecimal(noOfYear);
    BigDecimal totalValue = principal.add(principal.multiply(interest).multiply(noOfYearsBigDecimal));
    return totalValue;
    
  }
  
}

public class SimpleRunner {
  public static void main(String[] args) {
    Simple calculator = new Simple("4500.00", "7.5");
    BigDecimal totalValue = calculator.calculatorTotalValue(5);
    System.out.println(totalValue);
  }
}
