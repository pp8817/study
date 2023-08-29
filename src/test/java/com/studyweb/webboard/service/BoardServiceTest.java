package com.studyweb.webboard.service;

import com.studyweb.webboard.entity.Board;
import com.studyweb.webboard.repository.BoardRepository;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.Rollback;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Transactional
class BoardServiceTest {

    @Autowired
    private BoardRepository boardRepository;
    @Autowired
    private BoardService boardService;

    @Test
    @DisplayName("게시글 수정 테스트")
    void updateTest() {
        //given
        Board board = new Board("Spring", "Spring", "park");
        Board savedItem = boardService.save(board);
        Integer id = savedItem.getId();

        //when
        Board updateParam = new Board("java", "java", "sang");
        boardService.update(id, updateParam);

        //then
        Board findBoard = boardService.findById(id);
        assertThat(findBoard.getTitle()).isEqualTo(savedItem.getTitle());
        assertThat(findBoard.getContent()).isEqualTo(savedItem.getContent());
        assertThat(findBoard.getAuthor()).isEqualTo(savedItem.getAuthor());
        assertThat(findBoard).isEqualTo(savedItem);

    }

}