package com.studyweb.webboard.validation;

import com.studyweb.webboard.service.domain.board.Board;
import org.junit.jupiter.api.Test;

import javax.validation.ConstraintViolation;
import javax.validation.Validation;
import javax.validation.Validator;
import javax.validation.ValidatorFactory;
import java.util.Set;

public class BeanValidationTest {

    @Test
    void beanValidation() {
        ValidatorFactory factory = Validation.buildDefaultValidatorFactory();
        Validator validator = factory.getValidator();

        Board board = new Board();  
        board.setTitle(" "); //공백
        board.setAuthor(" ");
        board.setContent("테스트");

        Set<ConstraintViolation<Board>> violations = validator.validate(board);
        for (ConstraintViolation<Board> violation : violations) {
            System.out.println("violation = " + violation);
            System.out.println("violation.getMessage() = " + violation.getMessage());
        }


    }
}
