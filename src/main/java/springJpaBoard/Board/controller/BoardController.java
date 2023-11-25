package springJpaBoard.Board.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import springJpaBoard.Board.controller.requestdto.BoardForm;
import springJpaBoard.Board.controller.requestdto.CommentForm;
import springJpaBoard.Board.controller.requestdto.SaveCheck;
import springJpaBoard.Board.controller.requestdto.UpdateCheck;
import springJpaBoard.Board.controller.responsedto.BoardResponseDto;
import springJpaBoard.Board.controller.responsedto.CommentResponseDto;
import springJpaBoard.Board.controller.responsedto.MemberResponseDto;
import springJpaBoard.Board.domain.Board;
import springJpaBoard.Board.domain.Comment;
import springJpaBoard.Board.domain.Member;
import springJpaBoard.Board.domain.status.GenderStatus;
import springJpaBoard.Board.repository.search.BoardSearch;
import springJpaBoard.Board.service.BoardService;
import springJpaBoard.Board.service.CommentService;
import springJpaBoard.Board.service.MemberService;

import java.time.LocalDateTime;
import java.util.List;

import static java.util.stream.Collectors.toList;

@Controller
@RequiredArgsConstructor
@RequestMapping("/boards")
public class BoardController {

    private final BoardService boardService;
    private final MemberService memberService;
    private final CommentService commentService;

    /**
     * 게시글 작성
     */
    @GetMapping("/write")
    public String writeForm(Model model) {
        /**
         * 빈 껍데기인 MemberFrom 객체를 model에 담아서 가져가는 이유는 Validation의 기능을 사용하기 위해서이다.
         */
        List<Member> memberList = memberService.findMembers();

        List<MemberResponseDto> members = memberList.stream()
                .map(m -> new MemberResponseDto(m))
                .collect(toList());

        model.addAttribute("members", members);
        model.addAttribute("boardForm", new BoardForm());
        return "boards/writeBoardForm";
    }

    @PostMapping("/write")
    public String write(@Validated(SaveCheck.class) @ModelAttribute BoardForm boardForm, BindingResult result, @RequestParam("memberId") Long memberId) {

        /*
        오류 발생시(@Valid 에서 발생)
         */
        if (result.hasErrors()) {
            System.out.println("result = " + result.getAllErrors());
            return "boards/writeBoardForm";
        }
        Board board = new Board();
        board.createBoard(boardForm.getTitle(), boardForm.getContent(), boardForm.getWriter(), LocalDateTime.now());
        boardService.write(board, memberId);
        return "redirect:/";
    }

    /**
     * 게시글 목록
     */
//    @GetMapping
//    public String list(@RequestParam(value =  "offset", defaultValue = "0") int offset,
//                       @RequestParam(value = "limit", defaultValue = "100") int limit, Model model) {
//        List<Board> board = boardService.findBoardsMember(offset, limit);
//        List<BoardDto> boards = board.stream()
//                .map(b -> new BoardDto(b))
//                .collect(Collectors.toList());
//        model.addAttribute("boards", boards);
//
//        return "boards/boardList";
//    }
    @GetMapping
    public String boardList(@ModelAttribute("boardSearch") BoardSearch boardSearch, Model model, @PageableDefault(page = 0, size=2, sort = "id", direction = Sort.Direction.DESC) Pageable pageable) {

        Page<Board> boardList = null;

//        if (searchIsEmpty(boardSearch)) { //입력 X
//            boardList = boardService.boardList(pageable);
//        } else if (boardSearch.getMemberGender() == null) { //제목만 입력
//            boardList = boardService.search(boardSearch.getBoardTitle(), pageable);
//        } else { // 둘 다 입
//            boardList = boardService.searchGender(boardSearch.getBoardTitle(), boardSearch.getMemberGender(), pageable);
//        }

        if (searchIsEmpty(boardSearch)) {
            boardList = boardService.boardList(pageable);
        } else {
            String boardTitle = boardSearch.getBoardTitle();
            GenderStatus memberGender = boardSearch.getMemberGender();

            if (memberGender == null) {
                boardList = boardService.search(boardTitle, pageable);
            } else {
                boardList = boardService.searchGender(boardTitle, memberGender, pageable);
            }
        }

        model.addAttribute("boardPage", boardList);

        int nowPage = boardList.getPageable().getPageNumber() + 1;
        int startPage = Math.max(nowPage - 4, 1); //Math.max를 이용해서 start 페이지가 0이하로 되는 것을 방지
        int endPage = Math.min(nowPage + 5, boardList.getTotalPages()); //endPage가 총 페이지의 개수를 넘지 않도록

        List<BoardDto> boards = boardList.stream()
                .map(b -> new BoardDto(b))
                .collect(toList());

        model.addAttribute("boards", boards);
        model.addAttribute("nowPage", nowPage);
        model.addAttribute("startPage", startPage);
        model.addAttribute("endPage", endPage);

        return "boards/boardList";

    }

    private static boolean searchIsEmpty(BoardSearch boardSearch) {
        return (boardSearch.getBoardTitle() == "" || boardSearch.getBoardTitle() == null) && boardSearch.getMemberGender() == null;
    }

    /**
     * 게시글 상세 페이지
     */
    @GetMapping("/{boardId}/detail")
    public String detail(@PathVariable Long boardId, @PageableDefault(page = 0, size = 2, sort = "id",
            direction = Sort.Direction.DESC) Pageable pageable,Model model) {
        boardService.updateView(boardId); // views ++
        Board board = boardService.findOne(boardId); //이때 comments도 담아오게?
        BoardResponseDto boardDto = new BoardResponseDto(board);

        List<Member> memberList = memberService.findMembers();
        List<MemberResponseDto> members = memberList.stream()
                .map(m -> new MemberResponseDto(m))
                .collect(toList());

//        List<CommentResponseDto> comments = board.getComments();
//        List<CommentResponseDto> comments = board.getCommentList().stream()
//                .map(c -> new CommentResponseDto(c))
//                .collect(toList());
        Page<Comment> commentList = commentService.getCommentsByBno(boardId, pageable);
        List<CommentResponseDto> comments = commentList.stream()
                .map(c -> new CommentResponseDto(c))
                .collect(toList());

        /* 댓글 관련 */
        if (comments != null && !comments.isEmpty()) {
            model.addAttribute("comments", comments);
        }


        model.addAttribute("board", boardDto);
        model.addAttribute("members", members);
        model.addAttribute("commentForm", new CommentForm());

        return "boards/boardDetail";
    }

    /**
     * 게시글 수정
     */
    @GetMapping("/{boardId}/edit")
    public String updateBoardForm(@PathVariable("boardId") Long boardId, Model model) {
        Board board = boardService.findOne(boardId);

        BoardForm boardForm = new BoardForm();
        boardForm.createForm(board.getId(), board.getTitle(), board.getContent(), board.getWriter());

        model.addAttribute("boardForm", boardForm);
        return "/boards/updateBoardForm";
    }

    @PostMapping("/{boardId}/edit")
    public String updateBoard(@Validated(UpdateCheck.class) @ModelAttribute BoardForm boardForm, @PathVariable Long boardId) {

        /*
//        여기서 굳이 UpdateBoardDto로 변환을 해야할까? BoardForm과 UpdateBoardDto의 변수 차이는
         */
//        UpdateBoardDto boardDto = new UpdateBoardDto(boardId, boardForm.getTitle(), boardForm.getContent(), LocalDateTime.now());
        boardService.update(boardId, boardForm);

        return "redirect:/boards"; //게시글 수정 후 게시글 목록으로 이동
    }

    /**
     * 게시글 삭제
     */
    @GetMapping("/{boardId}/delete")
    public String deleteBoard(@PathVariable Long boardId){
        boardService.delete(boardId);

        return "redirect:/boards";
    }

    /**
     * BoardDto는 Controller에서만 사용하기 때문에 static class로 작성해줬다.
     */
    @Data
    static class BoardDto {
        private Long id;

        private String name;

        private String title;

        private String writer;

        private int view;

        private LocalDateTime boardDateTime;

        private int commentCount;

        public BoardDto(Board board) {
            this.id = board.getId();
            this.name = board.getMember().getName();
            this.title = board.getTitle();
            this.writer = board.getWriter();
            this.view = board.getView();
            this.boardDateTime = board.getBoardDateTime();
            this.commentCount = board.getCommentList().size();
        }
    }

}
