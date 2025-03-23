package jpabook.jpashop.repository.order.query;

import jpabook.jpashop.domain.Address;
import jpabook.jpashop.domain.OrderStatus;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class OrderFlatDto {
    //OrderQueryDto의 필드 변수
    private Long orderId;
    private String name;
    private LocalDateTime orderData;
    private Address address;
    private OrderStatus orderStatus;

    //OrderItemQueryDto의 필드 변수
    private String itemName;
    private int orderPrice;
    private int count;

    public OrderFlatDto(Long orderId, String name, LocalDateTime orderData, Address address, OrderStatus orderStatus, String itemName, int orderPrice, int count) {
        this.orderId = orderId;
        this.name = name;
        this.orderData = orderData;
        this.address = address;
        this.orderStatus = orderStatus;
        this.itemName = itemName;
        this.orderPrice = orderPrice;
        this.count = count;
    }
}
