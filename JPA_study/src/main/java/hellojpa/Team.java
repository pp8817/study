package hellojpa;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Entity
public class Team extends BaseEntity{

    @Id
    @GeneratedValue
    @Column(name = "TEAM_ID")
    private Long id;
    private String name;
    @OneToMany(mappedBy = "team") //mappedBy는 내가 뭐랑 연결되어 있는지를 알려주는 것. 연관관계의 주인이 아니면 모두 mappedBy로 주인 지정
    private List<Member> members = new ArrayList<>();

    public void addMember(Member member) { //연관관계 편의 메서드
//        member.setTeam(this);  //양쪽에 모두 값을 입력 - 순수한 객체 관계를 고려
//        members.add(member);
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<Member> getMembers() {
        return members;
    }

    public void setMembers(List<Member> members) {
        this.members = members;
    }
}