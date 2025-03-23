package com.chat.chatTest.domain.chat.repository;

import com.chat.chatTest.domain.chat.model.ChatMessage;
import com.chat.chatTest.domain.chat.model.ChatRoom;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    List<ChatMessage> findAllByChatRoom(ChatRoom chatRoom);
}
