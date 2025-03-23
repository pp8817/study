package com.studyweb.webboard.service;

import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

@SpringBootTest
@Transactional
class BoardServiceTest {

//    @Autowired
//    private BoardRepository boardRepository;
//    @Autowired
//    private BoardService boardService;
//    private MultipartFile file;
//
//    @Test
//    @DisplayName("게시글 수정 테스트")
//    void updateTest() throws Exception {
//
//        //given
//        Board board = new Board("Spring", "Spring", "park");
//        Board savedItem = boardService.save(board, file);
//        Integer id = savedItem.getId();
//
//        //when
//        Board updateParam = new Board("java", "java", "sang");
//        boardService.update(id, updateParam);
//
//        //then
//        Board findBoard = boardService.findById(id);
//        assertThat(findBoard.getTitle()).isEqualTo(savedItem.getTitle());
//        assertThat(findBoard.getContent()).isEqualTo(savedItem.getContent());
//        assertThat(findBoard.getAuthor()).isEqualTo(savedItem.getAuthor());
//        assertThat(findBoard).isEqualTo(savedItem);
//
//    }

}