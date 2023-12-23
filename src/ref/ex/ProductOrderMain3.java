package ref.ex;

import java.util.Scanner;

public class ProductOrderMain3 {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        System.out.println("입력할 주문의 개수를 입력하세요: ");
        int n = scanner.nextInt();
        scanner.nextLine(); // 입력 버퍼를 비우기 위한 코드

        // 여러 상품의 주문 정보를 담는 배열 생성
        ProductOrder[] orders = new ProductOrder[n];

        for (int i = 0; i < orders.length; i++) {
            System.out.println((i + 1) + "번째 주문 정보를 입력하세요.");

            System.out.print("상품명: ");
            String productName = scanner.nextLine();

            System.out.print("가격: ");
            int price = scanner.nextInt();

            System.out.print("수량: ");
            int quantity = scanner.nextInt();
            scanner.nextLine(); // 입력 버퍼를 비우기 위한 코드

            //createOrder()를 여러번 사용해서 상품 주문 정보들을 생성하고 배열에 저장
            orders[i] = createOrder(productName, price, quantity);
        }

        //printOrders()를 사용해서 상품 주문 정보 출력
        printOrders(orders);

        //getTotalAmount()를 사용해서 총 결제 금액 계산
        int totalAmount = getTotalAmount(orders);

        //총 결제 금액 출력
        System.out.println("총 결제 금액: " + totalAmount);
    }

    static ProductOrder createOrder(String productName, int price, int quantity) {
        ProductOrder productOrder = new ProductOrder();
        productOrder.productName = productName;
        productOrder.price = price;
        productOrder.quantity = quantity;
        return productOrder;
    }

    static void printOrders(ProductOrder[] orders) {
        for (ProductOrder o : orders) {
            System.out.println("상품명: " +o.productName+ ", 가격: " +o.price+ ", 수량:" + o.quantity);
        }
    }

    static int getTotalAmount(ProductOrder[] orders) {
        int totalAmount = 0;
        for (ProductOrder order : orders) {
            totalAmount += order.price*order.quantity;
        }

        return totalAmount;
    }
}
