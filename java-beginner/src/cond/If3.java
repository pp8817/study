package cond;

public class If3 {
    public static void main(String[] args) {
        int age = 14;

        if (age <= 7) {
            System.out.println("미취학");
        }
        if (age <= 17) {
            System.out.println("청소년");
        }
        if (age <= 27) {
            System.out.println("성인");
        }
    }
}
