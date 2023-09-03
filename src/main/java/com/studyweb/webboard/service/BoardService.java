package com.studyweb.webboard.service;

import com.studyweb.webboard.entity.Board;
import com.studyweb.webboard.repository.BoardRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class BoardService {
    private final BoardRepository boardRepository;

    public Board findById(Integer id) {
        return boardRepository.findById(id).get();
    }

    public Page<Board> boardList(Pageable pageable) {
        return boardRepository.findAll(pageable);
    }

    public Page<Board> boardSearchList(String searchKeyword, Pageable pageable) {
        return boardRepository.findByTitleContaining(searchKeyword, pageable);
    }

    public Board save(Board board) {
        return boardRepository.save(board);
    }

    public void delete(Integer id) {
        boardRepository.deleteById(id);
    }

    public void update(Integer id, Board updateBoard) {
        System.out.println("updateBoard = " + updateBoard);

        Board board = boardRepository.findById(id).get();
        board.setTitle(updateBoard.getTitle());
        board.setAuthor(updateBoard.getAuthor());
        board.setContent(updateBoard.getContent());

        boardRepository.save(board);
    }



}
