package com.studyweb.webboard.controller;

import com.studyweb.webboard.domain.Board;
import com.studyweb.webboard.file.FileStore;
import com.studyweb.webboard.service.BoardService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.util.UriUtils;

import java.net.MalformedURLException;
import java.nio.charset.StandardCharsets;

@Controller
@RequiredArgsConstructor
public class BoardController {

    private final BoardService boardService;
    private final FileStore fileStore;

    @GetMapping("/")
    public String mainPage() {
        return "main";
    }

    @GetMapping("/board/write")
    public String boardWriteForm(Model model) {
        model.addAttribute("post", new Board());
        return "boardWrite";
    }

    @PostMapping("/board/write")
    public String boardWrite(@ModelAttribute Board board, Model model, MultipartFile file) throws Exception {
        boardService.save(board, file);
        Integer id = board.getId();

//        model.addAttribute("localDateTime", LocalDateTime.now());
        model.addAttribute("message", "글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/post/" + id);

        return "message";
//        return "redirect:/board/post/" + id;
    }

    /**
     * 파일 업로드, 다운로드 컨트롤러
     */
    @ResponseBody
    @GetMapping("/images/{filename}")
    public Resource downloadImage(@PathVariable String filename) throws MalformedURLException {
        // "file:/Users/../0d713e88-4723-4088-bb4b-be039b6f9b47.png"
        //경로에 있는 파일에 접근해서 파일을 스트림?으로 반환을 함
        return new UrlResource("file:" + fileStore.getFullPath(filename));
    }

    @GetMapping("/attach/{postId}")
    public ResponseEntity<Resource> downloadAttach(@PathVariable Integer postId) throws MalformedURLException {
        Board post = boardService.findById(postId); //post를 접근할 수 있는 사용자만 사진 다운로드 가능
        String imageFilename = post.getFilename(); //사용자가 업로드한 파일 이름
        String imageFilepath = post.getFilepath(); //DB에 저장하는 파일 경로

        UrlResource resource = new UrlResource("file:" + fileStore.getFullPath(imageFilepath));

        String encodedUploadFileName = UriUtils.encode(imageFilename, StandardCharsets.UTF_8); // 한글, 특수문자가 깨지는 것을 방지
        String contentDisposition = "attachment; filename=\"" + encodedUploadFileName + "\"";

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, contentDisposition)
                .body(resource);
    }

    /****************/

    @GetMapping("/board/list")
    public String boardListForm(Model model, @PageableDefault(page = 0, size = 9, sort = "id",
            direction = Sort.Direction.DESC) Pageable pageable, String searchKeyword) {

        Page<Board> list = null;

        if (searchKeyword == null) {
            list = boardService.boardList(pageable); //검색을 안하면 현재 페이지 유지
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
        model.addAttribute("searchUrl", "/board/list"); //자동 리다이렉트

        return "message";
    }

    @GetMapping("/board/post/update/{id}")
    public String postUpdateForm(@PathVariable Integer id, Model model) {
        model.addAttribute("post", boardService.findById(id));
        return "boardUpdate";
    }

    @PostMapping("/board/post/update/{id}")
    public String postUpdate(@PathVariable Integer id, @ModelAttribute Board board,
                             Model model, MultipartFile file) throws Exception {

        boardService.update(id, board, file);

        model.addAttribute("message", "게시글 수정이 완료되었습니다.");
        model.addAttribute("searchUrl", "/board/post/" + id);

        return "message";
//        model.addAttribute("localDateTime", LocalDateTime.now());
//        return "redirect:/board/list";
    }
}