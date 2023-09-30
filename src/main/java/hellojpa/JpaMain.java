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
            Team team = new Team();
            team.setName("TeamA");
//            team.getMembers().add(member);
            em.persist(team);

            Member member = new Member();
            member.setUsername("member1");
            em.persist(member);

            team.addMember(member); //연관관계 편의 메서드

            em.flush();
            em.clear();

            Team findTeam = em.find(Team.class, team.getId());
            List<Member> members = findTeam.getMembers();


            System.out.println("===================");
            System.out.println("members = " + findTeam);
            System.out.println("===================");

            tx.commit(); // 트랜잭션을 커밋하는 시점에서 영속성 컨텍스트에 있는 DB의 쿼리가 날라감
        } catch (Exception e) {
            tx.rollback();
        } finally {
            em.close(); //중요, 엔티티 매니저가 결국 내부적으로 데이터베이스 커넥션을 물고 동작함. 사용 후 꼭 닫아줘야함.
        }
        emf.close();
    }
}
