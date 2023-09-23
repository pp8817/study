package com.studyweb.webboard.service.domain.board;

import com.studyweb.webboard.service.time.TimeEntity;
import lombok.Data;

import javax.persistence.*;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

@Entity
@Data
public class Board extends TimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @NotBlank(message = "공백x", groups = {SaveCheck.class, UpdateCheck.class})
//    @Size(min=2, max=40, groups = {SaveCheck.class, UpdateCheck.class})
    private String title;


    private String content;

    @NotBlank(groups = {SaveCheck.class})
//    @Size(max=10, groups = {SaveCheck.class})
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
