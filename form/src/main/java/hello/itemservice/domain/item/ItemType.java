package hello.itemservice.domain.item;

public enum ItemType {
    BOOK("도서"), FOOD("음식"), ETC("기타");

    private final String description; //상품 설명을 위해서 필드 추가

    ItemType(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
