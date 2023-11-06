package jpabook.jpashop.api;

import jpabook.jpashop.domain.Member;
import jpabook.jpashop.domain.service.MemberService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.stream.Collectors;

/**
 Request, Response DTO 중첩 클래스를 만들 때 static을 붙이는 이유
    1. 메모리 누수의 원인을 예방할 수 있고, 클래스의 각 인스턴스당 더 적은 메모리를 사용하게 된다.
    2. static이 아닌 멤버 클래스는 바깥 인스턴스와 암묵적으로 연결이 되기 때문에 바깥 인스턴스 없이는 생성할 수 없다.

    즉 멤버 클래스에서 바깥 인스턴스에 접근할 일이 없다면 무조건 static을 붙여 정적 멤버 클래스로 만들어주자!
 */
@RestController
@RequiredArgsConstructor
public class MemberApiController {

    private final MemberService memberService;

    /**
     *
     엔티티를 직접 노출하게 되면 엔티티에 있는 정보들이 다 외부에 노출이 된다.
     Json 요청시 반환하고 싶지 않은 변수, 값들에 @JsonIgnore를 사용하면 된다.
     하지만 API마다 요청이 다를 수 있기 때문에 이는 좋은 방식이 아니다.
     */
    @GetMapping("/api/v1/members")
    public List<Member> membersV1() {
        return memberService.findMembers();
    }

    @GetMapping("/api/v2/members")
    public Result memberV2() {
        List<Member> findMembers = memberService.findMembers();
        List<MemberDto> collect = findMembers.stream()
                .map(m -> new MemberDto(m.getName()))
                .collect(Collectors.toList());

        return new Result(collect);
    }


    /**
     *
     ※ Result 클래스로 컬렉션을 감싸는 이유는 Json 스펙에 맞춰 좀 더 유연하게 API를 반환하기 위함이다.

     만약 감싸지 않을 경우에는 리스트 자체로 [ {Data1}, {Data2}, {Data3}, ,,,] 이런 식으로 반환된다. 만약 API가 요구하는 것이 Data 뿐만 아니라 Data의 갯수도 반환을 요청했을 경우
     [ {Count}, {Data1}, {Data2}, {Data3} ,,] 이런 식으로 반환이 불가능하다. 리스트이기 때문에 Data와 Count객체가 동일해야한다.
     따라서, 좀 더 유연한 Json 객체 반환을 위해 { "count" : 3, "Data" : [ {Data1}, {Data2}, {Data3} ] } 이런 식으로 반환해주어야한다.
     그렇기 때문에 Result 클래스로 원하는 Json형태로 만들어 준 뒤 반환해주는 것이다.
     */
    @Data
    @AllArgsConstructor
    static class Result<T> {
//        private int count; //Result 클래스로 컬렉션을 감싸준다면 Data의 갯수를 반환해달라는 요청을 유연하게 수행 가능하다.
        private T data;
    }

    @Data
    @AllArgsConstructor
    static class MemberDto {
        private String name;

//        public MemberDto(String name) {
//            this.name = name;
//        }
    }

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
