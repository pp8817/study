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

            // 반환 타입이 명확할 때는 TypedQuery
            TypedQuery<Member> query = em.createQuery("select m from Member m", Member.class);

            Member result = em.createQuery("select m from Member m where m.username = :username", Member.class)
                    .setParameter("username", "member1")
                    .getSingleResult();
            System.out.println("singleResult = " + result.getUsername());

            // 반환 타입이 명확하지 않을 때는 Query
            Query query1 = em.createQuery("select m.username, m.age from Member m");

            List<Member> resultList = query.getResultList(); //컬렉션 반환
            Member result1 = query.getSingleResult(); //멤버 반환
            System.out.println("result1 = " + result1);

            for (Member member1 : resultList) {
                System.out.println("member1 = " + member1);
            }

            tx.commit();
        } catch (Exception e) {
            tx.rollback();
            e.printStackTrace();
        } finally {
            em.close();
        }

    }
}
