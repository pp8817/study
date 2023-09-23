package com.studyweb.webboard.service.domain.board;

import com.studyweb.webboard.service.domain.board.Board;
import com.studyweb.webboard.service.domain.UploadFile;
import com.studyweb.webboard.service.file.FileStore;
import com.studyweb.webboard.service.domain.board.BoardRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

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
            UploadFile uploadFile = fileStore.storeFile(file);
            board.setFilename(uploadFile.getUploadFilName());
//            board.setFilepath("/images/"+uploadFile.getserverFileName());
            board.setFilepath(uploadFile.getServerFileName());
        }

        boardRepository.save(board);
    }

    public void delete(Integer id) {
        boardRepository.deleteById(id);
    }

    public void update(Integer id, Board updateBoard, MultipartFile file) throws Exception {

        Board board = boardRepository.findById(id).get();
        board.update(updateBoard.getTitle(), updateBoard.getContent());

        if (!file.isEmpty()) {
            UploadFile uploadFile = fileStore.storeFile(file);

            board.setFilename(uploadFile.getUploadFilName());
            board.setFilepath(uploadFile.getServerFileName());
        }

        boardRepository.save(board);
    }

//    private static String getProjectPath() {
//        return System.getProperty("user.dir") + "/src/main/resources/static/files";
//    }


}