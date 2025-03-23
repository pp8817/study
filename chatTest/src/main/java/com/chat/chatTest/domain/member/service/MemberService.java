package com.chat.chatTest.domain.member.service;

import com.chat.chatTest.domain.member.dto.MemberDto;
import com.chat.chatTest.domain.member.model.Member;
import com.chat.chatTest.domain.member.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import static com.chat.chatTest.domain.member.dto.MemberDto.*;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class MemberService {
    private final MemberRepository memberRepository;

    @Transactional
    public void join(JoinMemberReq joinMemberReq) {
        memberRepository.save(joinMemberReq.toEntity());
    }

    public Member getMember(Long memberId) {
        return memberRepository.findById(memberId).get();
    }
}
