package com.studyweb.webboard.repository;

import com.studyweb.webboard.service.domain.Board;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

@SpringBootTest
@Transactional
class BoardRepositoryTest {

    @Autowired
    BoardRepository boardRepository;


    //수정 필요
    @Test
    void TimeEntity_등록() {
        //given
        LocalDateTime now = LocalDateTime.of(2023, 9, 10, 0, 0, 0);
        System.out.println("now = " + now);
        Board save = new Board();
        save.setTitle("title");
        save.setContent("content");
        save.setAuthor("author");

        boardRepository.save(save);

        System.out.println("save = " + save);


        //when
        List<Board> boardList = boardRepository.findAll();
        System.out.println("boardList = " + boardList);

        //then
        Board board = boardList.get(0);

        System.out.println(">>>>>>>>>>> createDate= " + board.getCreatedDate()
        +", modeifeidDate="+ board.getModifiedDate());

        assertThat(board.getCreatedDate()).isAfter(now);
        assertThat(board.getModifiedDate()).isAfter(now);
    }

}