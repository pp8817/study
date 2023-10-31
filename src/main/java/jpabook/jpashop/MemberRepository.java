package jpabook.jpashop;

import org.springframework.stereotype.Repository;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;

@Repository
public class MemberRepository {
    @PersistenceContext //엔티티 매니저, 팩토리를 알아서 생성해줌
    private EntityManager em;

    public Long save(Member member) {
        em.persist(member);
        return member.getId(); //커맨더랑 쿼리를 분리하기 위해 id 반환
    }

    public Member find(Long id) {
        return em.find(Member.class, id); //식별자 값에 해당하는 Member 객체를 찾아서 반환
    }

}
