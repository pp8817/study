package com.chat.chatTest.domain.chat.repository;

import com.chat.chatTest.domain.chat.model.ChatMessage;
import com.chat.chatTest.domain.chat.model.ChatRoom;
import com.chat.chatTest.domain.member.model.Member;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ChatRoomRepository extends JpaRepository<ChatRoom, Long> {
    Optional<ChatRoom> findBySenderAndReceiver(Member sender, Member receiver);
    Page<ChatRoom> findAllBySenderOrReceiver(Pageable pageable, Member sender, Member receiver);
}
