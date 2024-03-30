package com.example.WebSockterGuide.messagingstompwebsocket;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
public class HelloMessage {
    private String name;

    public HelloMessage(String name) {
        this.name = name;
    }

    public HelloMessage() {
    }
}
