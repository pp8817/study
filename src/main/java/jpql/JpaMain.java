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
            Team team = new Team();
            team.setName("teamA");
            em.persist(team);

            Member member = new Member();
            member.setUsername("관리자 1");
            member.setAge(10);
            member.setType(MemberType.ADMIN);

            member.setTeam(team);

            em.persist(member);

            Member member1 = new Member();
            member1.setUsername("관리자 2");
            member1.setAge(10);
            member1.setType(MemberType.ADMIN);

            member1.setTeam(team);

            em.persist(member1);


            em.flush();
            em.clear();

            String query3 = "select function('group_concat', m.username) From Member m";
            List<String> result3 = em.createQuery(query3, String.class)
                    .getResultList();
            for (String s : result3) {
                System.out.println("s = " + s);
            }
//
//            String query1 =
//                    "select " +
//                            "case when m.age <= 10 then '학생요금' " +
//                            "     when m.age >=60 then '경로요금' " +
//                            "     else '일반요금' " +
//                            "end " +
//                            "from Member m";
//            List<String> result = em.createQuery(query1, String.class)
//                    .getResultList();
//            for (String s : result) {
//                System.out.println("s = " + s);
//            }
//
//            String query2 = "select coalesce(m.username, '이름 없는 회원') as username " +
//                    "from Member m";
//            List<String> result2 = em.createQuery(query2, String.class)
//                    .getResultList();
//            for (String s : result2) {
//                System.out.println("s = " + s);
//            }
            em.createQuery("select m from Member m where m.username = :username", Member.class)
                    .setParameter("username", "member1")
                    .getSingleResult();

            tx.commit();
        } catch (Exception e) {
            tx.rollback();
            e.printStackTrace();
        } finally {
            em.close();
        }
    }
}
