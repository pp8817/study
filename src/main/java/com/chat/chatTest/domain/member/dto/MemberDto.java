package com.chat.chatTest.domain.member.dto;

import com.chat.chatTest.domain.member.model.Member;
import lombok.Builder;

public class MemberDto {

    /**
     * Request
     */
    @Builder
    public record JoinMemberReq(
            String loginId,
            String password
    ) {
        public Member toEntity() {
            return Member.builder()
                    .loginId(loginId)
                    .password(password)
                    .build();
        }
    }
}
