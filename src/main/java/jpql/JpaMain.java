package jpql;

import javax.persistence.*;
import java.util.List;

public class JpaMain {
    public static void main(String[] args) {
        EntityManagerFactory emf = Persistence.createEntityManagerFactory("hello");
        EntityManager em = emf.createEntityManager();

        EntityTransaction tx = em.getTransaction();
        tx.begin();

        try {
            Member member = new Member();
            member.setUsername("member1");
            member.setAge(10);
            em.persist(member);

            em.flush();
            em.clear();

            List<Member> result = em.createQuery("select m from Member m ", Member.class)
                    .getResultList(); // Entity들 반환, 반환된 엔티티들은 영속성 켄텍스트에서 관리가 될까?

            Member findMember = result.get(0);
            findMember.setAge(20); //Age가 20으로 변경된다면 영속성 켄텍스트에서 관리가 된다는 것.

            //연관관계인 Team 엔티티 가져오기
//            List<Team> resultTeam = em.createQuery("select m.team from Member m ", Team.class)
//                    .getResultList();
            //임베이드 타입 가져오기
//            em.createQuery("select o.address from Order o", Address.class)
//                    .getResultList();
            //스칼라 타입 가져오기
//            em.createQuery("select distinct m.username, m.age from Member m")
//                    .getResultList();
            List<MemberDTO> resultList = em.createQuery("select new jpql.MemberDTO(m.username, m.age) from Member m", MemberDTO.class)
                    .getResultList();

            MemberDTO memberDTO = resultList.get(0);
            System.out.println("memberDTO.getUsername() = " + memberDTO.getUsername());
            System.out.println("memberDTO.getAge() = " + memberDTO.getAge());

            tx.commit();
        } catch (Exception e) {
            tx.rollback();
            e.printStackTrace();
        } finally {
            em.close();
        }

    }
}
