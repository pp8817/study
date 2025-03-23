package com.securityproject.authenticationProject.users.service;

import com.securityproject.authenticationProject.users.domain.entity.Account;
import com.securityproject.authenticationProject.users.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;

    @Transactional
    public void create(Account account) {
        userRepository.save(account);
    }
}
