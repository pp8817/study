package com.example.WebSockterGuide.messagingstompwebsocket;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
public class Greeting {
    private String content;

    public Greeting(String content) {
        this.content = content;
    }

    public Greeting() {
    }
}
