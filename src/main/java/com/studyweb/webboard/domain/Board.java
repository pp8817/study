package com.studyweb.webboard.domain;

import com.studyweb.webboard.domain.time.TimeEntity;
import lombok.Data;
import java.util.List;

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

    private String filename;
    private String filepath;

    public void update(String title, String content) {
        this.title = title;
        this.content = content;
    }

    //기본 생성자 추가!
    public Board() {

    }
}
