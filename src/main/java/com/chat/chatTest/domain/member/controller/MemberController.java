package com.chat.chatTest.domain.member.controller;

import com.chat.chatTest.domain.member.dto.MemberDto;
import com.chat.chatTest.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import static com.chat.chatTest.domain.member.dto.MemberDto.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/member")
public class MemberController {
    private final MemberService memberService;

    @GetMapping
    public void joinMember(@RequestBody JoinMemberReq joinMemberReq) {
        memberService.join(joinMemberReq);
    }
}
