class NewExSub {
	public void sum(int a, int b){ // int a=0, int b=20;
		int sum=0, odd=0, even=0; // 초기값 설정
		for(int i=a; i<=b; i++){
			sum += i; // 총합 합계
			if(i%2==0){
				even += i; // 짝수합
			}else{
				odd += i; // 홀수합
			}
		}
		System.out.println(a+"~"+b+"까지의 합:"+sum);
		System.out.println(a+"~"+b+"까지의 짝수합:"+even);
		System.out.println(a+"~"+b+"까지의 홀수합:"+odd);
	}
}

public class NewEx01 {
	public static void main(String[] args) {
		NewExSub nes = null; // NewExSub 타입의 nes 객체 선언
		nes = new NewExSub(); 
        // new : 인스턴스 생성, Heap 메모리 공간 할당, 객체(nes)에게 참조값 리턴
		System.out.println(nes); // 출력 : @15db9742(참조값)
		nes.sum(0, 20); // 객체 참조값 이용 NewExSub의 sum메소드 호출 / 매게변수 전달
	}
}