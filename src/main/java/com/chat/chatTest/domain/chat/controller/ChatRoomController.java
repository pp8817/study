package com.chat.chatTest.domain.chat.controller;

import com.chat.chatTest.domain.chat.model.ChatRoom;
import com.chat.chatTest.domain.chat.service.ChatRoomService;
import com.chat.chatTest.domain.member.model.Member;
import com.chat.chatTest.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;


@RestController
@RequiredArgsConstructor
@RequestMapping("/chats")
public class ChatRoomController {
    private final MemberService memberService;
    private final ChatRoomService chatRoomService;

    // 채팅방 주소 가져오기
    @PostMapping("/create/{senderId}/{receiverId}")
    public ResponseEntity getOrCreateRoom(@PathVariable(name = "senderId") Long senderId, @PathVariable(name="receiverId") Long receiverId) {
        Member sender = memberService.getMember(senderId);
        Member reveiver = memberService.getMember(receiverId);

        Long roomId = chatRoomService.createRoom(sender, reveiver);

        URI location = UriComponentsBuilder.newInstance()
                .path("/chats/{room-id}")
                .buildAndExpand(roomId)
                .toUri();

        return ResponseEntity.created(location).build();
    }

    //  채팅방 열기
    @GetMapping("/{roomId}")
    public ResponseEntity getChatRoom(@PathVariable("roomId") long roomId,
                                      Member member) {
        ChatRoom chatRoom = chatRoomService.findRoom(roomId);

        return new ResponseEntity<>(chatRoom, HttpStatus.OK);
    }
}
