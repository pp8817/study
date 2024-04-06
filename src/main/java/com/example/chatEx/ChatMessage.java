package com.example.chatEx;

import lombok.Data;

@Data
public class ChatMessage {
    //메시지 타입
    public enum MessageType {
        ENTER, TALK
    }

    private MessageType type;
    private String roomId;
    private String sender;
    private String message;
}
