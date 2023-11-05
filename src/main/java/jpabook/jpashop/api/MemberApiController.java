package jpabook.jpashop.api;

import jpabook.jpashop.domain.Member;
import jpabook.jpashop.domain.service.MemberService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import javax.validation.constraints.NotEmpty;

@RestController
@RequiredArgsConstructor
public class MemberApiController {

    private final MemberService memberService;

    /**
     * api를 만들 때 파라미터로 엔티티를 받게 되면 엔티티 속 변수들의 이름이 변경된다면 (name -> username)
     * api 스펙이 변경된다.
     */
    @PostMapping("/api/v1/members")
    public CreateMemberResponse saveMember(@RequestBody @Valid Member member) {
        Long id = memberService.join(member);
        return new CreateMemberResponse(id);
    }

    /**
     * 파라미터로 엔티티가 아닌 DTO를 받자
     * 장점
     * 1. 엔티티에서 변수명이 변경된다면 setName에서 컴파일 오류가 발생한다.
     *   - setName만 수정하면 되기 때문에 api는 영향을 받지 않는다.
     * 2. DTO를 확인하면 API 스펙이 어떤 값을 받는지 바로 알 수 있다.
     *   - DTO에 Validation을 적용해도 된다.
     *     - 어떤 API 스펙에서는 Member의 이름이 공백이 가능하고, 어떤 API에서는 공백이 불가능할 수 있다.
     *     Member가 아닌 DTO에 @NotEmpty Validation을 적용한다면 위 문제를 해결할 수 있다.
     */
    @PostMapping("/api/v2/members")
    public CreateMemberResponse saveMember2(@RequestBody @Valid CreateMemberRequest request) {
        Member member = new Member();
        member.setName(request.getName());

        Long id = memberService.join(member);
        return new CreateMemberResponse(id);
    }

    @PutMapping("/api/v2/members/{id}")
    public UpdateMemberResponse updateMemberV2(@PathVariable("id") Long id,
                                               @RequestBody @Valid UpdateMemberRequest request) {
        memberService.update(id, request.getName());
        Member findMember = memberService.findOne(id);
        return new UpdateMemberResponse(findMember.getId(), findMember.getName());
    }

    @Data
    static class UpdateMemberRequest {
        private String name;
    }

    @Data
    @AllArgsConstructor
    static class UpdateMemberResponse {
        private Long id;
        private String name;
    }

    @Data
    static class CreateMemberRequest {
        @NotEmpty
        private String name;
    }

    @Data
    static class CreateMemberResponse {
        private Long id;

        public CreateMemberResponse(Long id) {
            this.id = id;
        }
    }

}
