package com.example.WebSockterGuide.messagingstompwebsocket;

import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration // Spring의 구성 클래스임을 나타냄
@EnableWebSocketMessageBroker // MessageBroker를 통해 WebSocket 메시지 처리 활성화
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        /*메시지 브로커를 구성하기 위해 WebSocketMessageBrokerConfigurer의 기본 메서드를 구현
        * 메시지 브로커는 메시지를 클라이언트 간의 라우팅 하는 역할을 한다.*/
        WebSocketMessageBrokerConfigurer.super.configureMessageBroker(registry);
    }

    @Override
    /*웹소켓 연결을 위한 /gs-guide-websocket (STOMP)엔드포인트를 등록*/
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        WebSocketMessageBrokerConfigurer.super.registerStompEndpoints(registry);
    }
}
