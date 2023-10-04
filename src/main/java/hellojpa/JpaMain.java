package hellojpa;

import hellojpa.mapp.Movie;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.Persistence;
import java.time.LocalDateTime;

public class JpaMain {
    public static void main(String[] args) {
        //db당 1개만 생성
        EntityManagerFactory emf = Persistence.createEntityManagerFactory("hello");
        //code, 쓰레드 간에 공유하면 안됨. 사용하고 버려야함
        EntityManager em = emf.createEntityManager();
        //JPA의 모든 데이터 변경은 트랜잭션 안에서 실행
        EntityTransaction tx = em.getTransaction();
        tx.begin();


        try {
            Member member = new Member();
            member.setUsername("hello");

            em.persist(member);

            em.flush(); //영속성 컨텍스트의 1차 캐시에 있는 정보를 db로 강제 전송
            em.clear(); //영속성 컨텍스트 초기화

            //
//            Member findMember = em.find(Member.class, member.getId());
//            System.out.println("findMember.getId() = " + findMember.getId());
//            System.out.println("findMember.getUsername() = " + findMember.getUsername());

            Member findMember = em.getReference(Member.class, member.getId()); //셀렉트 쿼리가 안나감
             System.out.println("findMember.getId() = " + findMember.getId());
             System.out.println("findMember.getUsername() = " + findMember.getUsername());
            tx.commit(); // 트랜잭션을 커밋하는 시점에서 영속성 컨텍스트에 있는 DB의 쿼리가 날라감
        } catch (Exception e) {
            tx.rollback();
        } finally {
            em.close(); //중요, 엔티티 매니저가 결국 내부적으로 데이터베이스 커넥션을 물고 동작함. 사용 후 꼭 닫아줘야함.
        }
        emf.close();
    }
}
