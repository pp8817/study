package jpabook.jpashop.domain.repository;

import jpabook.jpashop.domain.Member;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import javax.persistence.EntityManager;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class MemberRepository {
//    @PersistenceContext
    private final EntityManager em;

    public void save(Member member) {
        em.persist(member);
    }

    public Member findOne(Long id) {
        return em.find(Member.class, id); //단건 조회
    }

    public List<Member> findAll() {
        //JPQL과 SQL의 차이점: JPQL은 from의 대상이 테이블이 아닌 엔티티
        return em.createQuery("select m from Member m, Member.class") //JPQL 쿼리 사용
                .getResultList();
    }

    public List<Member> findByName(String name) { //이름으로 회원 조회
        return em.createQuery("select m from Member m where m.name = :name", Member.class) //파라미터 바인딩
                .setParameter("name", name)
                .getResultList();
    }
}
