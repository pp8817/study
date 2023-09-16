package com.studyweb.webboard.controller;

import com.studyweb.webboard.domain.Board;
import com.studyweb.webboard.service.BoardService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Controller
@RequiredArgsConstructor
public class BoardController {

    private final BoardService boardService;

    @GetMapping("/")
    public String mainPage() {
        return "main";
    }

    @GetMapping("/board/write")
    public String boardWriteForm() {
        return "boardWrite";
    }

    @PostMapping("/board/write")
    public String boardWrite(Board board, Model model, MultipartFile file) throws Exception {

        boardService.save(board, file);
//        model.addAttribute("localDateTime", LocalDateTime.now());
        model.addAttribute("message", "글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/list");

        return "message";
//        return "redirect:/board/list";
    }


    @GetMapping("/board/list")
    public String boardListForm(Model model, @PageableDefault(page = 0, size = 9, sort = "id",
            direction = Sort.Direction.DESC) Pageable pageable, String searchKeyword) {

        Page<Board> list = null;

        if (searchKeyword == null) {
            list = boardService.boardList(pageable);
        } else {
            list = boardService.boardSearchList(searchKeyword, pageable);
        }

        int nowPage = list.getPageable().getPageNumber() + 1;
        int startPage = Math.max(nowPage - 4, 1); //Math.max를 이용해서 start 페이지가 0이하로 되는 것을 방지
        int endPage = Math.min(nowPage + 5, list.getTotalPages()); //endPage가 총 페이지의 개수를 넘지 않도록

        model.addAttribute("list", list);
        model.addAttribute("nowPage", nowPage);
        model.addAttribute("startPage", startPage);
        model.addAttribute("endPage", endPage);


        return "boardList";
    }

    @GetMapping("/board/post/{id}")
    public String boardDetail(@PathVariable Integer id, Model model) {
        model.addAttribute("post", boardService.findById(id));
        return "boardDetail";
    }

    @GetMapping("/board/post/delete/{id}")
    public String postDelete(@PathVariable Integer id, Model model) {
        boardService.delete(id);

        model.addAttribute("message", "게시글이 삭제되었습니다.");
        model.addAttribute("searchUrl", "/board/list");

        return "message";
    }

    @GetMapping("/board/post/update/{id}")
    public String postUpdateForm(@PathVariable Integer id, Model model) {
        model.addAttribute("post", boardService.findById(id));
        return "boardUpdate";
    }

    @PostMapping("/board/post/update/{id}")
    public String postUpdate(@PathVariable Integer id, Board board, Model model, MultipartFile file) throws Exception {
        boardService.update(id, board, file);

        model.addAttribute("message", "게시글 수정이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/post/" + id);

        return "message";

//        model.addAttribute("localDateTime", LocalDateTime.now());

//        return "redirect:/board/list";
    }
}