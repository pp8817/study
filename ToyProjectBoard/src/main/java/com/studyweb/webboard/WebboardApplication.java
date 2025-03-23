package com.studyweb.webboard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

@SpringBootApplication
@EnableJpaAuditing
public class WebboardApplication {

	public static void main(String[] args) {
		SpringApplication.run(WebboardApplication.class, args);
	}

}
