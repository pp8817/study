package jpabook.jpashop.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jpabook.jpashop.domain.Member;
import lombok.Getter;
import lombok.Setter;
import org.aspectj.weaver.ast.Or;

import javax.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import static javax.persistence.FetchType.*;

@Entity
@Table(name = "orders") //order는 예약어로 지정되어 있을수가 있기 때문에 관례로 orders로 해준다.
@Getter
@Setter
public class Order {
    @Id
    @GeneratedValue
    @Column(name = "order_id")
    private Long id;

    @JsonIgnore
    @ManyToOne(fetch = LAZY)
    @JoinColumn(name = "member_id")
    private Member member; //연관관계의 주인

    /**
     * CascadeType.ALL은 OrderItem, Delivery를
     *    1. Order에서만 참조하고 있고
     *    2. Persist 해야하는 Life Cycle이 똑같이 때문에 CascadeType.ALL를 사용가능
     * 다른 곳에서도 '참조'한다면 해당 옵션은 사용 X, Order를 삭제할 때 같이 삭제된다면 참조가 걸려있는 다른 곳에서 문제가 발생.
     */
    @JsonIgnore
    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL)
    private List<OrderItem> orderItems = new ArrayList<>(); //연관관계 편의 메서드로 인해 주문한 상품들이 모여있음.

    @JsonIgnore
    @OneToOne(fetch = LAZY, cascade = CascadeType.ALL)
    @JoinColumn(name = "delivery_id")
    private Delivery delivery; //연관관계의 주인

    private LocalDateTime orderDate; //주문시간

    @Enumerated(EnumType.STRING)
    private OrderStatus status; //2가지 상태 ORDER, CANCEL

    //=====연관관계 편의 메서드=====// 위치는 핵심적으로 컨트롤하는 곳에
    public void setMember(Member member) {
        this.member = member;
        member.getOrders().add(this);
    }

    public void addOrderItem(OrderItem orderItem) {
        orderItems.add(orderItem);
        orderItem.setOrder(this);
    }

    public void setDelivery(Delivery delivery) {
        this.delivery = delivery;
        delivery.setOrder(this);
    }
    //==생성 메서드==//
    public static Order createOrder(Member member, Delivery delivery, OrderItem... orderItems) {
        Order order = new Order();

        /**
         * 연관 관계 편의 메서드로 양방향 연관관계 설정
         */
        order.setMember(member);
        order.setDelivery(delivery);

        for (OrderItem orderItem : orderItems) {
            order.addOrderItem(orderItem); //orderItems에 상품들을 더 넣어줌
        }

        order.setStatus(OrderStatus.ORDER);
        order.setOrderDate(LocalDateTime.now());
        return order; //order가 연관관계르 모두 걸면서 세팅이 되고 상태와 주문 시간 정보까지 다 세팅을 해서 반환된다.
    }

    //==비즈니스 로직==//
    /**
     * 주문 취소
     */
    public void cancel() {
        if (delivery.getStatus() == DeliveryStatus.COMP) { // 현재 배달 상태가 COMP(배달완료)면 에러 출력
            throw new IllegalStateException("이미 배송 완료된 상품은 취소가 불가능합니다.");
        }

        //현재 배달 상태가 COMP가 아니고 READY일 때
        this.setStatus(OrderStatus.CANCEL); // 주문 상태를 CANCEL로 변경하고

        for (OrderItem orderItem : this.orderItems) {
            // 주문 상품들을 하나씩 받아서 주문 취소한 상품 수량만큼 다시 상품 수량을 증가
            orderItem.cancel(); // getItem().addstock(count);
        }
    }

    //==조회 로직==//

    /**
     * 전체 주문 가격 조회
     */
    public int getTotalPrice() {
        int totalPrice = 0; //전체 상품 가굑
        for (OrderItem orderItem : this.orderItems) { //
            totalPrice += orderItem.getTotalPrice();
        }
        return totalPrice;
    }

}
