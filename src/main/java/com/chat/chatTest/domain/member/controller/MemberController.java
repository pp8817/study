package com.chat.chatTest.domain.member.controller;

import com.chat.chatTest.domain.member.dto.MemberDto;
import com.chat.chatTest.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import static com.chat.chatTest.domain.member.dto.MemberDto.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/member")
public class MemberController {
    private final MemberService memberService;

    @PostMapping
    public void joinMember(@RequestBody JoinMemberReq joinMemberReq) {
        memberService.join(joinMemberReq);
    }
}
