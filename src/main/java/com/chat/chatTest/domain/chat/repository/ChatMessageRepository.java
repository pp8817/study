package com.chat.chatTest.domain.chat.repository;

import com.chat.chatTest.domain.chat.model.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
}
