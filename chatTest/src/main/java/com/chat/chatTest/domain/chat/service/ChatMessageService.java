package com.chat.chatTest.domain.chat.service;

import com.chat.chatTest.domain.chat.dto.MessageDto;
import com.chat.chatTest.domain.chat.model.ChatMessage;
import com.chat.chatTest.domain.chat.model.ChatRoom;
import com.chat.chatTest.domain.chat.repository.ChatMessageRepository;
import com.chat.chatTest.domain.member.model.Member;
import com.chat.chatTest.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@Slf4j
@RequiredArgsConstructor
public class ChatMessageService {
    private final MemberService memberService;
    private final ChatRoomService roomService;
    private final ChatMessageRepository messageRepository;

    public void saveMessage(MessageDto dto, Long roomId) {
        Member member = memberService.getMember(dto.getSenderId());

        ChatRoom chatRoom = roomService.findRoom(roomId);

        ChatMessage chatMessage = ChatMessage.builder()
                .content(dto.getContent())
                .sender(member)
                .chatRoom(chatRoom)
                .build();

        messageRepository.save(chatMessage);
        log.info("메세지 저장 완료");
    }

    public List<ChatMessage> findMessages(long roomId, int page, int size) {
        ChatRoom chatRoom = roomService.findRoom(roomId);

        return messageRepository.findAllByChatRoom(chatRoom);
    }
}
