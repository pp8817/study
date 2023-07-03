package jpabook.jpashop.domain;

import lombok.Getter;
import lombok.Setter;

import javax.persistence.*;

import static javax.persistence.FetchType.*;

@Entity
@Getter
@Setter
public class Delivery {

    @Id
    @GeneratedValue
    @Column(name = "delivery_id")
    private Long id;
    @OneToOne(mappedBy = "delivery", fetch = LAZY)
    private Order order;

    @Embedded
    private Address address;
    @Enumerated(EnumType.STRING) //EnumType은 반드시 STRING로 해야함. 기본 값으로 하면 장애가 생김
    private DeliveryStatus status; //READY, COMP
}
