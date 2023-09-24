package hellojpa;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity //jpa가 로딩될 때 인식을 함
//@Table(name = "Member") //쿼리가 나갈 때 해당 테이블에 인서트하고 나감
public class Member {
    @Id //pk가 무엇인지 알려줌
    private Long id;

//    @Column(name = "name") // 컬럼 이름, 안적어주면 필드 명으로 지정
    private String name;

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
}
