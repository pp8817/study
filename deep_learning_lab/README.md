# 딥러닝응용2 수업 실습자료

## 실험 준비

1. cloning public lab materials
```bash
cd ~/
git clone https://github.com/Integrative-Data-Comprehension-Lab/Class_VisionAI_Lab_public.git
```

2. cloning your private repository
```bash
git clone https://YOUR_USERNAME:YOUR_TOKEN@github.com/\
YOUR_USERNAME/YOUR_PRIVATE_REPOSITORY_NAME.git
```

3. 실험 자료를 개인 레포지토리로 가져오기
```bash
cp Class_VisionAI_Lab_public/README.md YOUR_PRIVATE_REPOSITORY_NAME/
cp Class_VisionAI_Lab_public/lab_XX -r YOUR_PRIVATE_REPOSITORY_NAME/
```

4. 실험 진행 전 상태를 푸쉬하기
```bash
cd ~/YOUR_PRIVATE_REPOSITORY_NAME
git status
git add .
git commit -m "before lab_XX"
git push
```

## 과제 제출 방법
 - <mark>(주의)</mark> 폴더 구조, 파일 이름, 함수 이름, 또는 함수 인자를 변경할 경우 테스트 모듈이 정상 작동하지 않으니 주의할 것.
 - <mark>(주의)</mark> jupyter notebook에 테스트를 위해 기존에 없던 새로운 셀을 추가했다면 반드시 삭제할것. (코드 테스트가 실패할 수 있음)
 
1. .ipynb파일은 .py 파일로 변환한다.
``` bash
cd lab_XX
jupyter nbconvert FILE_NAME.ipynb --to script \
--TagRemovePreprocessor.enabled=True \
--TagRemovePreprocessor.remove_cell_tags execute_cell
```

2. 숙제 제출 전 코드를 최종 테스트 한다.
```bash
cd lab_XX
pytest
```

3. 깃 커밋 & 푸쉬 한다
```bash
# git add & commit (필요한 파일만 커밋할것)
git push
```