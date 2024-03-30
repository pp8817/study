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
    public void configureMessageBroker(MessageBrokerRegistry config) {
        /*메시지 브로커를 구성하기 위해 WebSocketMessageBrokerConfigurer의 기본 메서드를 구현
        * 메시지 브로커는 메시지를 클라이언트 간의 라우팅 하는 역할을 한다.*/

        // 메모리 기반 메시지 브로커가 '/topic' 접수사가 붙은 대상의 메시지를 클라이언트에게 전달
        config.enableSimpleBroker("/topic");
        //@MessageMapping으로 어노테이션이 달린 메서드에 바인딩된 메시지에 대해 /app 접두사를 지정
        //예를 들어 /app/hello는 GreetingController.greeting() 메서드가 처리하도록 매핑된 엔드포인트
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    /*웹소켓 연결을 위한 /gs-guide-websocket (STOMP)엔드포인트를 등록*/
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/gs-guide-websocket");
    }
}
