package hello.itemservice.domain.Item;

import hello.itemservice.domain.item.Item;
import hello.itemservice.domain.item.ItemRepository;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

public class ItemRepositoryTest {

    ItemRepository itemRepository = new ItemRepository();

//    @AfterEach
//    void afterEach() {
//        itemRepository.clearStore();
//    }

    @Test
    void save() {
        //given
        Item item = new Item("itemA", 10000, 10);
        //when
        Item savedItem = itemRepository.save(item);

        //then
        Item findItem = itemRepository.findById(savedItem.getId());

        assertThat(findItem).isSameAs(item);
    }
    @Test
    void findAll() {
        //given
        Item item1 = new Item("A", 10000, 1);
        Item item2 = new Item("B", 10000, 1);

        itemRepository.save(item1);
        itemRepository.save(item2);
        //when
        List<Item> findItems = itemRepository.findAll();

        //then
        assertThat(findItems.size()).isEqualTo(2);
        assertThat(findItems).contains(item1, item2);
    }

    @Test
    void updateItem() {
        //given
        Item item1 = new Item("A", 1000, 1);
        Item savedItem = itemRepository.save(item1);
        Long itemId = savedItem.getId();

        //when

        Item updateParam = new Item("B", 20000, 2);
        itemRepository.update(itemId, updateParam);


        //then
        Item findItem = itemRepository.findById(itemId);
        assertThat(findItem.getItemName()).isEqualTo(updateParam.getItemName());
        assertThat(findItem.getPrice()).isEqualTo(updateParam.getPrice());
        assertThat(findItem.getQuantity()).isEqualTo(updateParam.getQuantity());
    }
}
