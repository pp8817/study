package com.chat.chatTest.domain.chat.service;

import com.chat.chatTest.domain.chat.model.ChatRoom;
import com.chat.chatTest.domain.chat.repository.ChatRoomRepository;
import com.chat.chatTest.domain.member.model.Member;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Service
@Transactional
@RequiredArgsConstructor
public class ChatRoomService {
    private final ChatRoomRepository chatRoomRepository;

    public Long createRoom(Member sender, Member receiver) {
        // 둘의 채팅이 있는 지 확인
        Optional<ChatRoom> optionalChatRoom = chatRoomRepository.findBySenderAndReceiver(sender, receiver);
        Optional<ChatRoom> optionalChatRoom2 = chatRoomRepository.findBySenderAndReceiver(receiver, sender);

        ChatRoom chatRoom = null;

        if(optionalChatRoom.isPresent()) {
            chatRoom = optionalChatRoom.get();
            return chatRoom.getRoomId();
        } else if (optionalChatRoom2.isPresent()) {
            chatRoom = optionalChatRoom2.get();
            return chatRoom.getRoomId();
        } else {
            chatRoom = ChatRoom.builder().sender(sender).receiver(receiver).build();
        }

        ChatRoom saveChatRoom = chatRoomRepository.save(chatRoom);

        return saveChatRoom.getRoomId();
    }

    public ChatRoom findRoom(long roomId) {
        return chatRoomRepository.findById(roomId).get();
    }
}
