package jpabook.jpashop.domain.service;

import jpabook.jpashop.domain.item.Book;
import jpabook.jpashop.domain.item.Item;
import jpabook.jpashop.domain.repository.ItemRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class ItemService { //Service는 단순이 위임만 하는 클랴스이다.
    private final ItemRepository itemRepository;

    @Transactional
    public void saveItem(Item item) {
        itemRepository.save(item);
    }

    public List<Item> findItems() {
        return itemRepository.findAll();
    }

    public Item findOne(Long itemId) {
        return itemRepository.findOne(itemId);
    }

    public void updateItem(Long id, String name, int price, int stockQuantity) {
        Book book = (Book) itemRepository.findOne(id);
        book.setName(name);
        book.setPrice(price);
        book.setStockQuantity(stockQuantity);

        itemRepository.save(book);
    }
}
