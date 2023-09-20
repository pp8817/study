package com.studyweb.webboard.service.domain;

import com.studyweb.webboard.service.domain.time.TimeEntity;
import lombok.Data;

import javax.persistence.*;

@Entity
@Data
public class Board extends TimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String title;
    private String content;
    private String author;

//    private String attachName;
//    private String attachPath;

    private String filename; //시용자가 업로드한 파일 이름
    private String filepath; // DB에  저장되는 파일 이름(UUID 추가)

    public void update(String title, String content) {
        this.title = title;
        this.content = content;
    }

    //기본 생성자 추가!
    public Board() {

    }
}
