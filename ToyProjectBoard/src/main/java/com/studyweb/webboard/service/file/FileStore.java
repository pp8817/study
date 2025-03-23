package com.studyweb.webboard.service.file;

import com.studyweb.webboard.service.domain.UploadFile;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

@Component
public class FileStore {

    @Value("${file.dir}")
    private String fileDir;

    public String getFullPath(String filename) {
        return fileDir + filename;
    }

    /**
     * @param multipartFile
     * 추후 여러 장의 사진을 넣을 때 사용될 예정
     */
//    public List<UploadFile> storeFiles(List<MultipartFile> multipartFiles) throws IOException {
//        List<UploadFile> storeFileResult = new ArrayList<>();
//        for (MultipartFile multipartFile : multipartFiles) {
//            if (!multipartFile.isEmpty()) {
//                storeFileResult.add(storeFile(multipartFile)); //파일의 이름 정보가 들어간 UploadFile 객체를 storeFileResult에 넣어줌
//            }
//        }
//        return storeFileResult; //UploadFile 객체가 담긴 storeFileResult 반환
//
//    }

    public UploadFile storeFile(MultipartFile multipartFile) throws IOException {
        if (multipartFile.isEmpty()) {
            return null;
        }

        String originalFilename = multipartFile.getOriginalFilename();
        String serverFileName = createServerFileName(originalFilename); //랜덤의 uuid를 추가한 파일 이름
        multipartFile.transferTo(new File(getFullPath(serverFileName)));

        return new UploadFile(originalFilename, serverFileName);
    }

    // 서버 내부에서 관리하는 파일명은 유일한 이름을 생성하는 UUID를 사용해서 충돌하지 않도록 한다.
    private String createServerFileName(String originalFilename) {
        String ext = extractExt(originalFilename);
        String uuid = UUID.randomUUID().toString(); //파일 이름 중복 방지
        return uuid + "." + ext;
    }

    //확장자를 별도로 추출해서 서버 내부에서 관리하는 파일명에도 붙여준다.
    //Ex) a.png라는 이름으로 업로드하면 2def12-42qd-3214-e2dqda2.png 와 같이 확장자를 추가해서 저장한다.
    private String extractExt(String originalFilename) {
        int pos = originalFilename.lastIndexOf("."); //파일의 확장자 추출 ex) .png .img
        return originalFilename.substring(pos + 1);
    }
}
