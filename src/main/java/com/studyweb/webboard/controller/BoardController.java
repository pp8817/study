package com.studyweb.webboard.controller;

import com.studyweb.webboard.entity.Board;
import com.studyweb.webboard.service.BoardService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;

import java.time.LocalDateTime;

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
    public String boardWrite(Board board, Model model) {

        boardService.save(board);
//        model.addAttribute("localDateTime", LocalDateTime.now());
        model.addAttribute("message", "글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/list");

        return "message";
//        return "redirect:/board/list";
    }


    @GetMapping("/board/list")
    public String boardListForm(Model model) {
        model.addAttribute("list", boardService.boardList());
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
    public String postUpdate(@PathVariable Integer id, Board board, Model model) {
        boardService.update(id, board);

        model.addAttribute("message", "게시글 수정이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/post/" + id);

        return "message";

////        model.addAttribute("localDateTime", LocalDateTime.now());
//
//        return "redirect:/board/list";
    }
}
