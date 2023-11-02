package jpabook.jpashop.domain.service;

import jpabook.jpashop.domain.Delivery;
import jpabook.jpashop.domain.Member;
import jpabook.jpashop.domain.Order;
import jpabook.jpashop.domain.OrderItem;
import jpabook.jpashop.domain.item.Item;
import jpabook.jpashop.domain.repository.ItemRepository;
import jpabook.jpashop.domain.repository.MemberRepository;
import jpabook.jpashop.domain.repository.OrderRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class OrderService {
    private final OrderRepository orderRepository;
    private final MemberRepository memberRepository;
    private final ItemRepository itemRepository;

    /**
     * 주문
     */
    @Transactional
    public Long order(Long memberId, Long itemId, int count) {
        //엔티티 조회
        Member member = memberRepository.findOne(memberId);
        Item item = itemRepository.findOne(itemId);

        //배송정보 생성
        Delivery delivery = new Delivery();
        delivery.setAddress(member.getAddress()); // 배송 정보를 회원의 주소, 나중에는 배송지 정보를 따로 입력하도록 설정

        //주문상품 생성
        /**
         * 현재는 주문 상품이 1개만 넘어오도록 되어 있다.
         * 추후에 한 번에 여러개의 상품을 주문할 수 있도록 변경
         */
        OrderItem orderItem = OrderItem.createOrderItem(item, item.getPrice(), count); //상품 수량을 count만큼 줄임

        //주문 생성
        Order order = Order.createOrder(member, delivery, orderItem); //연관관계, 주문 상태, 주문 시간 세팅해서 order 객체로 반환

        /**
         * 원래 Delivery, OrderItem 모두 JPA에 persist하고 불러와서 사용을 해야한다.
         * 그러나 Order의 orderItems에 있는 CascadeType.ALL 옵션 덕분에 Order를 persist하면
         * CascadeType.ALL 옵션이 걸려있는 OrderItem과 Delivery도 persist를 날려준다.
         * 결론: order만 persist 해도 나머지도 엮여서 persist 된다.
         */
        //주문 저장
        orderRepository.save(order); //이때 Delivery, OrderItem도 같이 Persist

        return order.getId();
    }

    /**
     * 주문 취소
     */
    @Transactional
    public void cancel(Long orderId) {
        //주문 엔티티 조화
        Order order = orderRepository.findOne(orderId);
        //주문 취소
        order.cancel();
    }

    //검색
//    public List<Order> findOrders(OrderSearch orderSearch) {
//        return orderRepository.findAll(orderSearch);
//    }
}
