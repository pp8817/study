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
            Address address = new Address("city", "street", "1000");

            Member member = new Member();
            member.setUsername("member1");
            member.setHomeAddress(address);
            em.persist(member);

            Address copyAddress = new Address(address.getCity(), address.getStreet(), address.getZipcode());

            Member member2 = new Member();
            member2.setUsername("member2");
            member2.setHomeAddress(copyAddress);
            em.persist(member2);

            member.getHomeAddress().setCity("newCity");

            tx.commit(); // 트랜잭션을 커밋하는 시점에서 영속성 컨텍스트에 있는 DB의 쿼리가 날라감
        } catch (Exception e) {
            tx.rollback();
            e.printStackTrace();
        } finally {
            em.close(); //중요, 엔티티 매니저가 결국 내부적으로 데이터베이스 커넥션을 물고 동작함. 사용 후 꼭 닫아줘야함.
        }
        emf.close();
    }
}
