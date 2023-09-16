package com.studyweb.webboard.domain;

import lombok.Data;

@Data
public class UploadFile {
    private String uploadFilName;
    private String storeFileName;

    public UploadFile(String uploadFilName, String storeFileName) {
        this.uploadFilName = uploadFilName;
        this.storeFileName = storeFileName;
    }
}
