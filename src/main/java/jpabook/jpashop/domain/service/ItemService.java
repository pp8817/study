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

    @Transactional
    public void updateItem(UpdateItemDto itemDto) {
        Item findItem = itemRepository.findOne(itemDto.getId());
        /**
         * Dirty Checking 발생, 가능하다면 setter는 사용하지 않는 방법으로 구현
         */
//        findItem.setName(name);
//        findItem.setPrice(price);
//        findItem.setStockQuantity(stockQuantity);
        findItem.modifyItem(itemDto.getName(), itemDto.getPrice(), itemDto.getStockQuantity());
    }
}
