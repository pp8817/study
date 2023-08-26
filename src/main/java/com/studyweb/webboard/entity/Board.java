package com.studyweb.webboard.entity;

import lombok.Data;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity
@Data
public class Board {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String title;
    private String content;
    private String author;

    public Board(String title, String content, String author) {
        this.title = title;
        this.content = content;
        this.author = author;
    }

    //기본 생성자 추!
    public Board() {

    }
}
