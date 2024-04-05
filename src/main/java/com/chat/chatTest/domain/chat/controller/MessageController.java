package com.chat.chatTest.domain.chat.controller;

import com.chat.chatTest.domain.chat.service.ChatMessageService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class MessageController {
    private final ChatMessageService chatMessageService;
    private final ChannelTo
}
