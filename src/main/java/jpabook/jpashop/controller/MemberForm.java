package jpabook.jpashop.controller;

import lombok.Getter;
import lombok.Setter;

import javax.validation.constraints.NotEmpty;

@Getter
@Setter
/**
 * 회원 가입에서 사용할 값을 받는 객체, DTO
 */
public class MemberForm {

    @NotEmpty(message = "회원 이름은 필수입니다.") //Validation
    private String name;

    private String city;
    private String street;
    private String zipcode;
}
