package com.studyweb.webboard.service;

import com.studyweb.webboard.entity.Board;
import com.studyweb.webboard.repository.BoardRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

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

    public void save(Board board, MultipartFile file) throws Exception {

        //저장할 경로를 지정
        String projectPath = getProjectPath();

        // 랜덤 식별자 생성
        UUID uuid = UUID.randomUUID();

        // uuid + _ + 파일의 원래 이름
        String fileName = uuid + "_" + file.getOriginalFilename();

        //파일을 생성할 것인데 경로는 projectPath, 이름은 filename로 담긴다는 뜻
        File saveFile = new File(projectPath, fileName);

        file.transferTo(saveFile);

        board.setFilename(fileName); //DB에 filename 저장
        board.setFilepath("/files/" + fileName);

        boardRepository.save(board);
    }

    public void delete(Integer id) {
        boardRepository.deleteById(id);
    }

    public void update(Integer id, Board updateBoard, MultipartFile file) throws Exception {

        Board board = boardRepository.findById(id).get();
        board.update(updateBoard.getTitle(), updateBoard.getContent());

        if (file.isEmpty()) {
            System.out.println("filename = " + board.getFilename());
            System.out.println("Filepath = " + board.getFilepath());

            boardRepository.save(board);

        } else {
            //저장할 경로를 지정
            String projectPath = getProjectPath();

            // 랜덤 식별자 생성
            UUID uuid = UUID.randomUUID();

            // uuid + _ + 파일의 원래 이름
            String fileName = uuid + "_" + file.getOriginalFilename();

            //파일을 생성할 것인데 경로는 projectPath, 이름은 filename로 담긴다는 뜻
            File saveFile = new File(projectPath, fileName);

            file.transferTo(saveFile);

            board.setFilename(fileName); //DB에 filename 저장
            board.setFilepath("/files/" + fileName);

            boardRepository.save(board);
        }
    }

    private static String getProjectPath() {
        return System.getProperty("user.dir") + "/src/main/resources/static/files";
    }


}
