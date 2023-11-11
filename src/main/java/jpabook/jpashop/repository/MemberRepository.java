package jpabook.jpashop.repository;

import jpabook.jpashop.domain.Member;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MemberRepository extends JpaRepository<Member, Long> {

    /**
     이것 이상의 구현이 필요하지 않다. findByName을 실행하면 Spring Data JPA가
     "select m from Member m where m.name = :name"라고 쿼리문을 작성해준다.
     */
    //
    List<Member> findByName(String name);
}
