package com.studyweb.webboard.entity;

import lombok.Builder;
import lombok.Data;

import javax.persistence.*;

@Entity
@Data
public class Board extends TimeEntity{

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String title;
    private String content;
    private String author;

    private String filename;
    private String filepath;


    //수정을 위한 생성자
    @Builder
    public Board(String title, String content, String author) {
        this.title = title;
        this.content = content;
        this.author = author;
    }

    public void update(String title, String content) {
        this.title = title;
        this.content = content;
    }

    //기본 생성자 추가!
    public Board() {

    }
}
