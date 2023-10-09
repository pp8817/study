package jpabook.jpashop.domain;
import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Entity
public class Member extends BaseEntity{

    @Id
    @GeneratedValue
    @Column(name = "MEMBER_ID")
    private Long Id;
    private String name;

    @Embedded
    private Address address;

    @OneToMany(mappedBy = "member") //양방향 매핑, 연관관계 주인 설
    private List<Order> orders = new ArrayList<>(); // 초기 값을 new ArrayList<>();로 주는 것은 보통 관계로 많이 사용

    public Long getId() {
        return Id;
    }

    public void setId(Long id) {
        Id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Address getAddress() {
        return address;
    }

    public void setAddress(Address address) {
        this.address = address;
    }

    public List<Order> getOrders() {
        return orders;
    }

    public void setOrders(List<Order> orders) {
        this.orders = orders;
    }
}
