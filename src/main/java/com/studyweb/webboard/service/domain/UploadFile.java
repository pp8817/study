package com.studyweb.webboard.service.domain;

import lombok.Data;

@Data
//@Getter, @Setter, @ToString, @EqualsAndHashCode, @RequiredArgsConstructor
public class UploadFile {
    private String uploadFilName; //사용자가 업로드한 파일 이름
    private String serverFileName; //DB에 저장되는 파일 이름(중복 방지를 위해 UUID 추가)

    public UploadFile(String uploadFilName, String serverFileName) {
        this.uploadFilName = uploadFilName;
        this.serverFileName = serverFileName;
    }
}
