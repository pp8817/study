package hellojpa;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.Persistence;
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
//            Member findMember = em.find(Member.class, 1L);
//            findMember.setName("HelloJPA"); //수정의 경우에는 persistence 할 필요 X,

//            List<Member> result = em.createQuery("select m from Member as m", Member.class)
//                    .getResultList(); //멤버 객체를 대상으로 쿼리, 객체가 대상
//
//            for (Member member : result) {
//                System.out.println("member = " + member);
//            }
            //비영속
//            Member member = new Member();
//            member.setId(101L);
//            member.setName("HelloJPA");

            //영속
//            em.persist(member); //이때 db에 저장되는 것은 아님.
            Member findMember1 = em.find(Member.class, 101L);//조회를 할 때 셀렉트 쿼리가 안나감. 이유? 1차 캐시에 저장이 됐기 때문에 pk 값으로 가져옴
            Member findMember2 = em.find(Member.class, 101L); //이때는 쿼리가 안나감

//            em.detach(member); 영속성 컨텍스트에서 지워버리는 
            tx.commit(); // 트랜잭션을 커밋하는 시점에서 영속성 컨텍스트에 있는 DB의 쿼리가 날라감
        } catch (Exception e) {
            tx.rollback();
        } finally {
            em.close(); //중요, 엔티티 매니저가 결국 내부적으로 데이터베이스 커넥션을 물고 동작함. 사용 후 꼭 닫아줘야함.
        }

        emf.close();

    }
}
