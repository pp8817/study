package jpabook.jpashop.api;

import jpabook.jpashop.domain.Address;
import jpabook.jpashop.domain.Order;
import jpabook.jpashop.domain.OrderStatus;
import jpabook.jpashop.repository.OrderRepository;
import jpabook.jpashop.repository.OrderSearch;
import jpabook.jpashop.repository.order.simplequery.OrderSimpleQueryDto2;
import jpabook.jpashop.repository.order.simplequery.OrderSimpleQueryRepository;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * xToOne(ManyToOne, OneToOne)
 * Order
 * Order -> Member
 * Order -> Delivery
 */
@RestController
@RequiredArgsConstructor
public class OrderSimpleApiController {

    private final OrderRepository orderRepository;
    private final OrderSimpleQueryRepository orderSimpleQueryRepository;

    /**
     * 무한 루프에 빠지게 됨.
     * 문제를 해결하려면 양방향 연관관계가 걸려있는 곳에 모두 @JsonIgnore을 해야함.
     */
    @GetMapping("/api/v1/simple-orders")
    public List<Order> ordersV1() {
        List<Order> all = orderRepository.findAllByCriteria(new OrderSearch());
        for (Order order : all) {
            order.getMember().getName(); //Lazy 강제 초기화
            order.getDelivery().getAddress(); //Lazy 강제 초기화
        }
        return all;
    }

    @GetMapping("/api/v2/simple-orders")
    public Result ordersV2() {
        /**
         * ORDER 2개 조회
         * 1+N 문제: 1+ 회원 N + 배송 N 문제 -> 총 5번의 쿼리가 실행, 조회되는 ORDER의 수가 늘어날수록 쿼리는 기하급수적으로 증가
         */
        List<Order> orders = orderRepository.findAllByCriteria(new OrderSearch());
        List<SimpleOrderDto> result = orders.stream()
                .map(o -> new SimpleOrderDto(o)) //DTO로 맵핑
                .collect(Collectors.toList()); //LIST로 변환
        return new Result(result); //Result로 감싸주기
    }

    @GetMapping("/api/v3/simple-orders")
    public Result orderV3() {
        /**
         fetch join을 사용해서 쿼리가 1번 나감. V2에 비해서 쿼리문 갯수가 많이 감소.
         */
        List<Order> orders = orderRepository.findAllWithMemberDelivery();
        List<SimpleOrderDto> result = orders.stream()
                .map(o -> new SimpleOrderDto(o))
                .collect(Collectors.toList());

        return new Result(result);
    }

    @GetMapping("/api/v4/simple-orders")
    public Result orderV4() {
        /**
         JPA에서 DTO로 바로 조회
         원하는 것만 셀렉트, 최적화 가능
         But 재사용성이 떨어짐
         orderRepository에서 DTO를 조회하면 API가 리포지토리에 들어와 있는 것과 마찬가지고
         리포지토리의 순수성이 깨지게 됨.
         해결법: 성능 최적화된 쿼리용 리포지토리를 별도로 뽑는다.
         */
//        List<OrderSimpleQueryDto> orderDtos = orderRepository.findOrderDtos();
//        return new Result(orderDtos);

        List<OrderSimpleQueryDto2> new_orderDtos = orderSimpleQueryRepository.findOrderDtos();
        return new Result(new_orderDtos);
    }

    @Data
    static class SimpleOrderDto {
        private Long orderId;
        private String name;
        private LocalDateTime orderDate;
        private OrderStatus orderStatus;
        private Address address;

        public SimpleOrderDto(Order order) {
            this.orderId = order.getId();
            this.name = order.getMember().getName(); //Lazy 초기화
            this.orderDate = order.getOrderDate();
            this.orderStatus = order.getStatus();
            this.address = order.getDelivery().getAddress(); //Lazy 초기화
        }
    }

    @Data
    @AllArgsConstructor
    static class Result<T> {
        private T data;

//        public Result(T data) {
//            this.data = data;
//        }
    }

}
