package jpabook.jpashop.api;

import jpabook.jpashop.domain.Address;
import jpabook.jpashop.domain.Order;
import jpabook.jpashop.domain.OrderItem;
import jpabook.jpashop.domain.OrderStatus;
import jpabook.jpashop.repository.OrderRepository;
import jpabook.jpashop.repository.OrderSearch;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.*;

@RestController
@RequiredArgsConstructor
public class OrderApiController {

    private final OrderRepository orderRepository;

    /**
     * V1: 엔티티 직접 노출
     * - 엔티티가 변하면 API 스펙이 변한다.
     * - 트랜잭션 안에서 지연 로딩 필요
     * - 양방향 연관관계 문제
     */
    @GetMapping("/api/v1/orders")
    public List<Order> ordersV1() {
        List<Order> all = orderRepository.findAllByCriteria(new OrderSearch());
        /**
         Lazy 로딩 초기화
         */
        for (Order order : all) {
            order.getMember().getName();
            order.getDelivery().getAddress();

            List<OrderItem> orderItems = order.getOrderItems();
            orderItems.stream().forEach(o -> o.getItem().getName());
        }
        return all;
    }

    /**
     * V2: 엔티티를 조회해서 DTO로 반환(fecth join 사용X)
     * 트랜잭션 안에서 지연 로딩 필요
     * <p>
     * 쿼리가 너무 많이 나간다. 1+N+N+N 번
     */
    @GetMapping("/api/v2/orders")
    public Result ordersV2() {
        List<Order> orders = orderRepository.findAllByCriteria(new OrderSearch());
        List<OrderDto> collect = orders.stream()
                .map(o -> new OrderDto(o))
                .collect(toList());
        return new Result(collect);
    }


    /**
     * V3: 엔티티를 조회해서 DTO로 변환(fetch join 사용O)
     * - 페이징 시에는 N 부분을 포기해야함(대신에 batch fetch size? 옵션 주면 N -> 1 쿼리로 변경 가능)
     * <p>
     * 단점: 페이징이 불가능함. V3.1에서 개선
     * - 일대다에서 일(1)을 기준으로 페이징을 하는 것이 목적이다. 그런데 데이터는 다(N)를 기준으로 row가 생성된다.
     * - Order를 기준으로 페이징 하고 싶은데, 다(N)인 OrderItem을 조인하면 OrderItem이 기준이 되어버린다.
     * <p>
     * 컬렉션 fetch join은 사용하지 말자!
     */
    @GetMapping("/api/v3/orders")
    public Result OrderV3() {
        List<Order> orders = orderRepository.findAllWithItem();
        List<OrderDto> collect = orders.stream()
                .map(o -> new OrderDto(o))
                .collect(toList());

        return new Result(collect);
    }

    /**
     * V3에서 페이징이 불가능하다는 문제가 있었다.
     * 페이징 + 컬렉션 엔티티를 함께 조회하려면 어떻게 해야할까?
     * 대부분의 페이징 + 컬렉션 엔티티 조회 문제는 아래 방법으로 해결할 수 있다.

     먼저 ToOne 관계를 모두 페치 조인한다. ToOne 관계는 row수를 증가시키지 않으므로 페이징 쿼리에 영향을 주지 않는다.
     컬렉션은 지연 로딩으로 조회한다.
     지연 로딩 성능 최적화를 위해 hibernate.default_batch_fetch_size, @BatchSize 를 적용한다.
     - hibernate.default_batch_fetch_size: 글로벌 설정
     - @BatchSize: 개별 최적화
     */
    @GetMapping("/api/v3.1/orders")
    public Result OrderV3_page(@RequestParam(value = "offset", defaultValue = "0") int offset,
                               @RequestParam(value = "limit", defaultValue = "100") int limit) {
        List<Order> orders = orderRepository.findAllWithMemberDelivery(offset, limit);

        List<OrderDto> collect = orders.stream()
                .map(o -> new OrderDto(o))
                .collect(toList());
        return new Result(collect);
    }



    @Data
    @AllArgsConstructor
    static class Result<T> {
        private T data;

//        public Result(T data) {
//            this.data = data;
//        }
    }

    /**
     * OrderItem 또한 엔티티이기 때문에 노출하면 안된다.
     * Address는 밸류 오브젝트이고 바뀔 일이 없기 때문에 상관없다.
     */
    @Data
    static class OrderDto {

        private Long orderId;
        private String name;
        private LocalDateTime orderDate;
        private OrderStatus orderStatus;
        private Address address;
        private List<OrderItemDto> orderItems;

        public OrderDto(Order order) {
            this.orderId = order.getId();
            this.name = order.getMember().getName();
            this.orderDate = order.getOrderDate();
            this.orderStatus = order.getStatus();
            this.address = order.getDelivery().getAddress();
            this.orderItems = order.getOrderItems().stream()
                    .map(orderItem -> new OrderItemDto(orderItem))
                    .collect(toList());
        }
    }

    /**
     * OrderDto를 만들 때 OrderItem Entity의 데이터도 끌고 온다.
     * 이때 Order만 Dto로 감싸주는 것이 아니라 OrderItem 또한 Dto로 감싸서
     * 엔티티에 의존하는 것을 모두 없애야한다.
     */
    @Getter
    static class OrderItemDto {
        private String itemName; //상품 명
        private int orderPrice; //주문 가격
        private int count; //주문 수량

        public OrderItemDto(OrderItem orderItem) {
            this.itemName = orderItem.getItem().getName();
            this.orderPrice = orderItem.getOrderPrice();
            this.count = orderItem.getCount();
        }
    }
}
