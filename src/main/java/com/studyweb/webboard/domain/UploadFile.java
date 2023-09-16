package com.studyweb.webboard.domain;

import lombok.Data;

@Data
public class UploadFile {
    private String uploadFilName; //사용자가 업로드한 파일 이름
    private String storeFileName; //DB에 저장되는 파일 이름(중복 방지를 위해 UUID 추가)

    public UploadFile(String uploadFilName, String storeFileName) {
        this.uploadFilName = uploadFilName;
        this.storeFileName = storeFileName;
    }
}
