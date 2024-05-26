package com.securityproject.authenticationProject.security.service;

import com.securityproject.authenticationProject.users.domain.dto.AccountContext;
import com.securityproject.authenticationProject.users.domain.dto.AccountDto;
import com.securityproject.authenticationProject.users.domain.entity.Account;
import com.securityproject.authenticationProject.users.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.modelmapper.ModelMapper;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;

@Service("userDetailsService")
@RequiredArgsConstructor
public class FormUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        Account account = userRepository.findByUsername(username);

        if (account == null) { // 회원 정보가 없다면 예외 발생
            throw new UsernameNotFoundException("No user found with username" + username);
        }

        // 회원의 권한 정보를 가져와서 List 타입으로 저장
        List<GrantedAuthority> authorities = List.of(new SimpleGrantedAuthority(account.getRoles()));
        ModelMapper mapper = new ModelMapper();
        AccountDto accountDto = mapper.map(account, AccountDto.class);

        return new AccountContext(accountDto, authorities); // UserDetails 타입으로 반환
    }
}
