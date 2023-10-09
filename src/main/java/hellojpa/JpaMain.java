package hellojpa;

import hellojpa.mapp.Movie;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.Persistence;
import java.time.LocalDateTime;
import java.util.List;

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
            member.setUsername("member1");
            member.setHomeAddress(new Address("homeCity", "street", "1000"));

            member.getFavoriteFoods().add("치킨");
            member.getFavoriteFoods().add("피자");
            member.getFavoriteFoods().add("족발");

            member.getAddressesHistory().add(new AddressEntity("old1", "street", "1000"));
            member.getAddressesHistory().add(new AddressEntity("old2", "street", "1000"));

            em.persist(member);

            em.flush();
            em.clear();

            System.out.println("============== START ===============");
            Member findMember = em.find(Member.class, member.getId());

            //homeCicy -> newCity
//            findMember.getHomeAddress().setCity("newCity");

            Address oldAddress = findMember.getHomeAddress();
            findMember.setHomeAddress(new Address("newCicy", oldAddress.getStreet(), oldAddress.getZipcode()));

            //치킨 -> 한식
            findMember.getFavoriteFoods().remove("치킨");
            findMember.getFavoriteFoods().add("한식");

            findMember.getAddressesHistory().remove(new AddressEntity("old1", "street", "1000"));
            findMember.getAddressesHistory().add(new AddressEntity("newCity1", "street", "1000"));

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
