package com.securityproject.authenticationProject.users.repository;

import com.securityproject.authenticationProject.users.domain.entity.Account;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<Account, Long> {
    Account findByUsername(String username);
}
