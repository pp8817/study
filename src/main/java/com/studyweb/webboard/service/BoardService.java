package com.studyweb.webboard.service;

import com.studyweb.webboard.domain.Board;
import com.studyweb.webboard.domain.UploadFile;
import com.studyweb.webboard.file.FileStore;
import com.studyweb.webboard.repository.BoardRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class BoardService {
    private final BoardRepository boardRepository;
    private final FileStore fileStore;

    @Value("${file.dir}")
    private String fileDir;

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

        if(!file.isEmpty()){ //파일 업로드가 있는 경우에만 실행
            //저장할 경로를 지정
//        String projectPath = fileDir;

            // 랜덤 식별자 생성
//            UUID uuid = UUID.randomUUID();
//
//            // uuid + _ + 파일의 원래 이름
//            String fileName = uuid + "_" + file.getOriginalFilename();
//
//            //파일을 생성할 것인데 경로는 projectPath, 이름은 filename로 담긴다는 뜻
//            File saveFile = new File(fileDir, fileName);
//
//            file.transferTo(saveFile); //파일 저장
//
//            board.setFilename(fileName); //DB에 filename 저장
//            board.setFilepath("/files/" + fileName);
            UploadFile uploadFile = fileStore.storeFile(file);
            board.setFilename(uploadFile.getUploadFilName());
            board.setFilepath("/files/"+uploadFile.getStoreFileName());
        }


        boardRepository.save(board);
    }

    public void delete(Integer id) {
        boardRepository.deleteById(id);
    }

    public void update(Integer id, Board updateBoard, MultipartFile file) throws Exception {

        Board board = boardRepository.findById(id).get();
        board.update(updateBoard.getTitle(), updateBoard.getContent());

        if (file.isEmpty()) {

            boardRepository.save(board);

        } else {
            //저장할 경로를 지정
//            String projectPath = fileDir;

            // 랜덤 식별자 생성
            UUID uuid = UUID.randomUUID();

            // uuid + _ + 파일의 원래 이름
            String fileName = uuid + "_" + file.getOriginalFilename();

            //파일을 생성할 것인데 경로는 projectPath, 이름은 filename로 담긴다는 뜻
            File saveFile = new File(fileDir, fileName);

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