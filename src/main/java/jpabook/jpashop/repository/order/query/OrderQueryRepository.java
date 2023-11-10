package jpabook.jpashop.repository.order.query;

import jpabook.jpashop.api.OrderApiController;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;

import javax.persistence.EntityManager;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * OrderRepository의 경우 Order Entity를 조회하는 용도로 사용하는 것이고
 * OrderQueryRepository는 화면이나 API에 의존 관계가 있는 부분을 떼어내는 용도이다.서

 핵심 비즈니스를 위한 엔티티를 찾을 때 -> OrderRepository
 엔티티가 아닌 화면에 핏한 쿼리들을 이용할 때 -> OrderQueryRepository
   - 관심사의 분리, 생명주기가 다르기 때문
 */
@Repository
@RequiredArgsConstructor
public class OrderQueryRepository {

    private final EntityManager em;

    /**
     컬렉션은 별도로 조회
     Query: 루트 1번, 컬렉션 N 번제 -> 1+N 문제가 생
     Query: 루트 1번, 컬렉션 N 번
     단건 조회에서 많이 사용하는 방식
     */
    public List<OrderQueryDto> findOrderQueryDtos() {
        //루트 조회(toOne 코드를 모두 한번에 조회) - 컬렉션을 제외한 나머지 한 번에 조회

        List<OrderQueryDto> result = findOrders(); //query 1번 -> N번

        //루프를 돌면서 컬렉션 추가(추가 쿼리 실행) - 1:N 관계인 컬렉션 조회(orderItems)
        result.forEach(o -> {
            List<OrderItemQueryDto> orderItems = findOrderItems(o.getOrderId()); //더미 데이터의 경우 한 번에 데이터가 2개씩 넘어옴
            o.setOrderItems(orderItems);
        });
        return result;
    }

    /**
     최적화
     Query: 루트 1번, 컬렉션 1번 - 총 2번
     데이터를 한번에 처리할 때 많이 사용하는 방식
     */
    public List<OrderQueryDto> findAllDto_optimization() {
        //루트 조회(toOne 코드를 모두 한번에 조회), 쿼리 1번
        List<OrderQueryDto> result = findOrders(); //Order를 Dto로 가져오면서 컬렉션을 제외한 Member, Delivery 엔티티 ToOne 관계를 강제로 영속성 컨텍스트로 올림

        /*
        findOrders로 조회한 Order의 Dto들을 id로 변환 -> OrderId의 List
        더미 데이터의 경우 orderId: 4, 11
         */
        List<Long> orderIds = result.stream()
                .map(o -> o.getOrderId())
                .collect(Collectors.toList());

        //쿼리 1번
        List<OrderItemQueryDto> orderItems = em.createQuery(
                "select new jpabook.jpashop.repository.order.query.OrderItemQueryDto(oi.order.id, i.name, oi.orderPrice, oi.count) " +
                        "from OrderItem oi " +
                        "join oi.item i " +
                        "where oi.order.id in :orderIds", OrderItemQueryDto.class)
                .setParameter("orderIds", orderIds)
                .getResultList();

        //lambda식을 활용해서 List -> Map로 최적화
        // Key: OrderID, Value: OrderItemQueryDto List
        Map<Long, List<OrderItemQueryDto>> orderItemMap = orderItems.stream()
                .collect(Collectors.groupingBy(orderItemQueryDto -> orderItemQueryDto.getOrderId()));

        /**
         * 루프를 돌면서 컬렉션 추가(추가 쿼리 실행X)

         Query를 한 번 날리고 메모리에서 Map로 가져온 다음 메모리에서 매칭을 해서 값을 세팅
         */
        result.forEach(o -> o.setOrderItems(orderItemMap.get(o.getOrderId())));

        return result;
    }

    ///////////////////// Dto에서 사용되는 Query 조회 로직 ////////////////////

    /**
     * 1:N 관계(컬렉션)을 제외한 나머지를 한 번에 조회
     */
    public List<OrderQueryDto> findOrders() {
        return em.createQuery("select new jpabook.jpashop.repository.order.query.OrderQueryDto(o.id, m.name, o.orderDate, o.status, d.address)" +
                        " from Order o " +
                        "join o.member m " +
                        "join o.delivery d", OrderQueryDto.class)
                .getResultList();
    }

    /**
     * 1:N 관계인 orderItems 조회
     */
    private List<OrderItemQueryDto> findOrderItems(Long orderId) {
        return em.createQuery(
                        "select  new jpabook.jpashop.repository.order.query.OrderItemQueryDto(oi.order.id, i.name, oi.orderPrice, oi.count)" +
                                "from OrderItem oi " +
                                "join oi.item i " +
                                "where oi.order.id = :orderId", OrderItemQueryDto.class) //OrderId로 구분
                .setParameter("orderId", orderId)
                .getResultList();
    }

}
