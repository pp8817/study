#!/usr/bin/env python
# coding: utf-8

# # Basics of Python

# 이 자료는 `CS231n` Python tutorial (http://cs231n.github.io/python-numpy-tutorial/) by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335)를 적절하게 수정한 것이다.

# 파이썬은 널리 쓰이는 general-purpose programming language로, 다른 유명한 라이브러리 (numpy, scipy, matplotlib)와 함께 데이터 컴퓨팅 분야에서 매우 널리 쓰인고 있따.

# 이 튜토리얼에서는 아래 내용들을 다룬다.
# 
# * 파이썬 기본: 기본 데이터 타입 (Containers, Lists, Dictionaries, Sets, Tuples), 함수, 클래스
# * Numpy: Arrays, Array indexing, Datatypes, Array math, Broadcasting
# * Matplotlib: Plotting, Subplots, Images
# * IPython: Creating notebooks, Typical workflows

# ## Jupyter Notebook 사용하기

# Jupyter notebook은 여러개의 cell로 구성되어 있으며 각각의 cell 은 파이썬 코드나 설명을 담을 수 있다.
# 
# cell은 크게 `Code` 와 `Markdown` 두가지 종류로 나뉘어지며, 현재 설명하고 있는 이 cell은 `Markdown` cell이다
# 
# cell을 실행하려면 원하는 cell을 선택한 뒤 `Shift-Enter`를 입력한다
# 
# 만약 `Code` cell이라면 코드가 실행될 것이며 그 결과가 cell아래에 출력된다.
# 
# 만약 `Markdown` cell이라면 내용이 적절하기 표기된다.

# 아래는 `Code` cell이다.

# In[ ]:


x = 1
print(x)


# Cell간에는 전역 변수들이 공유된다.

# In[ ]:


y = 2 * x
print(y)


# ### Keyboard Shortcuts
# 
# * `esc`: cell 편집을 종료 
# * 화살표를 통해 cell간에 이동할 수 있다.
# * `b`: 현재 cell 아래에 새로운 cell 삽입
# * `a`: 현재 cell 위에 새로운 cell 삽입
# * `dd`: cell 삭제
# * `m` : cell을 `Markdown`으로 변경. `esc` 모드에 있어야 한다.
# * `y` : cell을 `Markdown`으로 변경. `Code` 모드에 있어야 한다.
# 

# `Kernel -> Restart & Clear Output`을 통해 파이썬 커널을 재시작하거나, 코드 실행결과를 모두 삭제할 수 있다.
# 
# 재시작후에는 모든 이전 작업이 삭제되므로 처음부터 다시 cell을 실행하여야 한다.

# ## 파이썬 기본

# 파이썬은 high-level 프로그래밍 언어로, 변수는 동적으로 타이핑 된다 (dynamic typing).
# 
# 파이썬은 보통 pseudocode와 같다고 불릴 정도로 아주 복잡한 작업을 단지 몇줄의 코드로 작성할 수 있다.
# 
# 아래는 그 예시로 quicksort 알고리즘의 구현이다.

# In[ ]:


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))


# ### Python versions

# 파이썬은 크게 2.x and 3.x으로 나뉘어지며, Python 3.0 이 소개되면서 2.x로 작성된 코드를 지원하지 않는 큰 변화가 일어났다. 
# 
# 딥러닝과 머신러닝을 위해서는 이 강좌에서는 Python 3.8 이상을 사용하는것을 추천한다.

# ### Basic data types

# #### 숫자형 (Numeric)

# 정수(Integers) 실수(floats), 복소수 (Complex number)에 대한 사칙연산이 잘 정의되어 있다.

# In[ ]:


x = 3
print(x, type(x))

x = 5.0
print(x, type(x))

x = 3 + 4j
print(x, type(x))


# In[ ]:


x = 5.0
print(x + 1)   # Addition;
print(x - 1)   # Subtraction;
print(x * 2)  # Multiplication;
print(x ** 2)  # Exponentiation;


# In[ ]:


x += 1 # x = x + 1을 의미
print(x) 
x *= 2 # x = x * 1을 의미
print(x) 


# #### 불리언(Booleans)

# True / False의 값을 갖는 불리언 연산은 다른 프로그래밍 언어에서 (`&&`, `||`, etc.)를 사용하는것과는 달리 영어 단어를 사용한다.

# In[ ]:


t, f = True, False
print(type(t)) # Prints "<type 'bool'>"


# In[ ]:


print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;


# #### 문자열(Strings)

# In[ ]:


hello_var = 'hello'   # 문자열은 single quotes
world_var = "world"   # 또는 double quotes로 표현되며 무엇을 사용하든 좋다.
print(hello_var)
print(len(world_var)) # 문자열의 길이를 반환한다.


# In[ ]:


hw = hello_var + ' ' + world_var  # 문자열의 덧셈은 string concatenation으로 정의된다.
print(hw)


# In[ ]:


hw12 = f'{hello_var}_{world_var}_{x}' # f-string을 통해 문자열과 다른 형식의 데이터를 편리하게 출력할수 있다.
print(hw12)  


# 문자열을 다루는 다양한 메서드들이 정의되어 있다.

# In[ ]:


s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"


# ### 컨테이너 (Containers)

# 컨테이너는 자료를 담는 그릇이라는 뜻으로, 파이썬에는, lists, dictionaries, sets, tuples이 정의되어 있다.

# #### 리스트(Lists)

# 리스트는 순서를 가지고 중복을 허용하는 객체의 집합이며, 크기를 변경할 수도 있고 다른 타입의 데이터를 저장할 수도 있다.

# In[ ]:


xs = [3, 1, 2]   # Create a list
print(xs)
print(len(xs)) #리스트의 길이를 가져온다.
print(xs[1])     # 리스트 접근에는 [] 연산자를 사용한다. index = 1 (2번째 변수) 에 접근. index는 0부터 시작한다.
print(xs[-1])    # 음수 index는 리스트 끝에서 부터 접근한다.


# In[ ]:


xs[2] = 'foo'    # 리스트는 다른 형식의 자료를 담을 수 있다.
print(xs)


# In[ ]:


xs.append('bar') # 리스트 끝에 새로운 자료를 추가할 수 있다.
print(xs)  


# In[ ]:


x = xs.pop()    # pop()은 마지막 element를 꺼내고 반환한다.
print(x, xs)


# In[ ]:


xs += ["Abi", "Kevin"] #리스트 간의 덧셈은 레스트를 concatenate하는것이다
print(xs)


# 빈 리스트 만들기

# In[ ]:


more_names = []
more_names = list()


# ##### 슬라이싱(Slicing)

# 파이썬은 슬라이싱이라고 불리는 sublist에 접근할 수 있는 손쉬운 방법을 제공한다.

# In[ ]:


a = ['a', 'b', 'c', 'd', 'e']
print(a)        
print(a[2:])    # index 2부터 끝까지 가져온다.
print(a[:2])    # 처음부터 index 2까지 가져온다 (2는 제외)
print(a[2:4])   # index 2 부터 4까지 가져온다 (4는 제외)
print(a[:])     # 전체 리스트를 가져온다.
print(a[:-1])   # 음수 인덱스도 가능하다.

a[2:4] = [8, 9] # sublist에 새로운 값을 할당한다.
print(a)        # Prints "[0, 1, 8, 9, 4]"


# ##### 루프 (Loops)

# 리스트와 같이 순회가능한(iterable)객체는 for를 이용하여 순회할 수 있다.

# In[ ]:


animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)


# index도 같이 얻고싶다면 `enumerate`를 사용하면 된다.

# In[ ]:


animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print(f'index {idx}: {animal}')


# ##### 리스트 컴프리헨션(List comprehensions)

# 아래와 같이 리스트를 순회하며 거듭제곱을 하는 코드를 생각해보자.

# In[ ]:


nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


# 리스트 컴프리헨션을 사용하면 이 코드를 훨씬 가독성있고 간단하게 만들 수 있다.

# In[ ]:


nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)


# 리스트 컴프리헨션은 조건문을 이용할 수도 있다.

# In[ ]:


nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]  # x가 짝수일 경우에만
print(even_squares)


# #### 딕셔너리(Dictionaries)

# 딕셔너리는 사전이라는 뜻으로 딕셔너리 자료형은 (key, value) 쌍을 가지는 자료형이다.
# 리스트와 같이 순차적으로(sequential) 요소를 가져오는것이 아니라 key값을 이용해 value값을 얻어온다.

# In[ ]:


d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])      # key = 'cat'에 해당하는 value를 가져옴
print('cat' in d)     # 딕셔너리가 특정 key를 갖고있는지 확인


# In[ ]:


d['fish'] = 'wet'    # 새로운 (key,value)쌍을 추가함
print(d)    


# In[ ]:


print(d['monkey'])  #  'monkey'가 key에 없으면 KeyError를 반환


# In[ ]:


print(d.get('monkey', 'N/A'))  # key에 해당하는 값이 있으면 value를 반환, 없으면 default값을 반환
print(d.get('fish', 'N/A'))


# In[ ]:


del d['fish']        # 딕셔너리에서 쌍 삭제하기
print(d.get('fish', 'N/A')) # "fish"는 더이상 없다.


# 딕셔너리도 이터러블(iterable)이다

# In[ ]:


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    print(f'A {animal} has {d[animal]} legs')


# (key, value) 쌍에 접근하고 싶으면 ``items``를 사용한다

# In[ ]:


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print(f'A {animal} has {legs} legs')


# 리스트와 비슷하게 딕셔너리도 딕셔너리 컴프리헨션을 통해 생성할 수 있다.

# In[ ]:


nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# #### 집합(Sets)

# 집합은 순서가 없고 반복이 불가능한 요소들의 모임이다.

# In[ ]:


animals = {'cat', 'dog'}
print('cat' in animals)  # Check if an element is in a set
print('fish' in animals) 


# In[ ]:


animals.add('fish')      # 새로운 요소를 집합에 추가
print('fish' in animals)
print(len(animals))       # Number of elements in a set;


# In[ ]:


animals.add('cat')       # 이미 있는 요소를 추가하면 아무런 일이 일어나지 않는다.
print(len(animals))       
animals.remove('cat')    # 요소 삭제
print(len(animals))       


# 집합도 iterable이다. 하지만 집합에는 순서가 없어서 순서를 예측할 수 없다.

# In[ ]:


animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"


# Set comprehensions: 리스트, 딕셔너리와 동일하다.

# In[ ]:


from math import sqrt
print({int(sqrt(x)) for x in range(30)})


# #### 튜플(Tuples)

# 튜플은 순서가 있고 불변하는(immutable) 값들의 리스트이다. 튜플은 리스트와 비슷하지만 가장 큰 차이중 하나는 딕셔너리의 key로 사용되거나 집합의 요소로 수 있다는 것이다.

# In[ ]:


d = {(x, x + 1): x for x in range(10)}  # 튜플을 키로 가지는 딕셔너리
print(d)


# In[ ]:


t = (5, 6)       # 튜플 만들기
print(type(t))
print(d[t])       
print(d[(1, 2)])


# In[ ]:


t[0] = 1 #튜플의 값은 변경할 수 없다 (immutable)


# #### mutable과 immutable
# 파이썬은 모든것이 객체인데, 그 객체들은 값이 변경가능한 mutable과 불가능한 immutable로 나뉘어진다.
# - immutable : 숫자(number), 문자열(string), 튜플(tuple)
# - Mutable : 리스트(list), 집합(set), 딕셔너리(dictionary)

# mutable객체는 그 안에 담겨 있는 값이 다른 함수에 의해 변경될 수 있으니 주의해야 한다.

# In[ ]:


x = [1,2,3,4]
y = x            # 리스트는 shallow copy된다.
x.append(5)

print(x)
print(y)


# 실제 값까지 복사 (deep copy)를 위해서는 ``copy``함수를 이용한다

# In[ ]:


x = [1,2,3,4]
y = x.copy()            # deep copy
x.append(5)

print(x)
print(y)


# ### 함수(Functions)

# 파이썬 함수는 `def` 키워드를 이용해 정의한다.

# In[ ]:


def get_sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(get_sign(x)) # 함수 호출에는 괄호 ()를 사용한다.


# 함수의 인자 (argument)로 기본값을 기정할 수 있다.

# In[ ]:


def hello(name, loud=False):
    if loud:
        print(f'HELLO, {name.upper()}')
    else:
        print(f'Hello, {name}')
        
hello('Bob')
hello('Fred', loud=True)


# 인자가 많은 함수의 경우 가변인자 (variable argument)를 이용할 수 있다

# In[ ]:


def sum_numbers(*args):
    print(type(args)) #tuple로 전달받는다.

    return sum(args)

print(sum_numbers(10, 15, 20))


# print함수는 사실 가변인자를 이용하여 구현된다

# In[ ]:


print("My", "name", "is")


# 많은 인자를 전달하면서 인자의 이름을 같이 전달하고 싶을 경우 키워드 인자(keyword argument)를 사용한다

# In[ ]:


def greet_me(**kwargs):
    print(type(kwargs)) #딕셔너리 형식으로 전달된다
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 함수 호출
greet_me(name="민준", message="안녕하세요") 


# 딕셔너리를 이용해 인자를 전달할 수도 있다

# In[ ]:


def greet_me(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

d = {str(x) : x for x in range(10)}
greet_me(**d) 


# 결합해서 사용할 수도 있다.

# In[ ]:


def create_user_profile(name, email, *args, **kwargs):
    profile = {
        "name": name,
        "email": email,
        "interests": args,
        "additional_info": kwargs
    }
    return profile

# 함수 호출 예시
profile = create_user_profile("김지민", "jimin@example.com", "여행", "독서", age=29, city="서울")
print(profile)


# ### 클래스(Classes)

# In[ ]:


class Greeter:

    # 생성자 (constructor)
    def __init__(self, name):
        self.name = name  # 새로운 인스턴스를 생성하는 경우 처음에만 실행된다.

    # 인스턴스 메서드
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Greeter 클래스의 새로운 인스턴스(instance)를 생성
g.greet()            # 인스턴스 메서드 호출; prints "Hello, Fred"
g.greet(loud=True)


# ### 매직메서드와 파이썬 객체
# 파이썬은 모든 것이 객체이며, 내가 구현한 클래스에 적절한 메직메서드를 구현함으로써 파이썬 문법과 잘 결합할 수 있다.
# |
# 컨테이너는 ``__contains__()`` 매직메서드를 구현한 객체이다. 이 매직메서드는 ``in`` 키워드가 발견될때 호출된다

# In[ ]:


class Boundaries:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __contains__(self, coord):
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height
    
boundary = Boundaries(10, 5)
print((2,3) in boundary)
print((5,10) in boundary)


# 이터러블: ``__iter__`` 매직 메서드를 구현한 객체
# 
# 이터레이터: ``__next__`` 매직 메서드를 구현한 객체

# In[ ]:


from datetime import date, timedelta

class DateRangeIterable:
    """ 자체 이터레이터 메서드를 가지고 있는 이터러블"""

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._present_day = start_date

    def __iter__(self):
        return self

    def __next__(self):
        if self._present_day >= self.end_date:
            raise StopIteration()
        today = self._present_day
        self._present_day += timedelta(days=1)
        return today


# For loop의 작동원리는 ``StopIteration`` 예외가 작성할 때 까지 ``__next__()``를 호출하는 것이다.

# In[ ]:


for day in DateRangeIterable(date(2022, 1, 1), date(2022, 1, 5)):
    print(day)


# callable (호출가능한) 객체는 ``__call__`` 메직메서드를 구현한 클래스이다.
# 
# 

# In[ ]:


class ValueAccumulator:
    def __init__(self):
        self._counts = 0
    def __call__(self, val): 
        self._counts += val
        return self._counts


# In[ ]:


va = ValueAccumulator() 
print(va(3))
print(va(5))


# ## Numpy

# Numpy는 scientific computing에서 아주 중요한 라이브러리로 다차원 배열 계산에 아주 높은 성능을 보여준다.

# In[ ]:


import numpy as np


# ### Arrays
# 
# numpy array 는 행렬 혹은 grid형태로 값이 배열된것이며, 모두 같은 데이터 타입을 가진다.

# In[ ]:


a = np.array([1, 2, 3])  # 1차원 array 생성
print(type(a))
print(a.shape)           # array의 차원을 tuple형태로 가져온다


# In[ ]:


print(a[0], a[1], a[2])  # 요소 접근
a[0] = 5                 # 값 변경
print(a)                  


# In[ ]:


b = np.array([[1,2,3],[4,5,6]])   # 2차원 array 생성
print(b)
print(b.shape) 
print(b[0, 0], b[0, 1], b[1, 0])  # 요소 접근


# array를 생성하는 다양한 함수들이 제공된다

# In[ ]:


a = np.zeros((2,2))  # Create an array of all zeros with shape (2, 2)
print(a)


# In[ ]:


b = np.ones((1,2))   # Create an array of all ones
print(b)


# In[ ]:


c = np.full((2,2), 7) # Create a constant array
print(c)


# In[ ]:


d = np.eye(2)        # Create a 2x2 identity matrix
print(d)


# In[ ]:


e = np.random.random((2,2)) # Create an array filled with random values
print(e)


# In[ ]:


f = np.arange(10)
print(f)


# ### Array indexing
# 
# 파이썬 리스트와 동일하게 다양한 indexing방식이 제공된다
# 
# Slicing도 가능하며 다차원배열일 경우 각 차원에 대해 slicing을 명시해야한다.

# In[ ]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

b = a[:2, 1:3]
print(b)


# numpy array도 mutable이며 slicing한 데이터를 수정할 경우 원본 array도 같이 변경된다.

# In[ ]:


print(a[0, 1])  
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a) 


# In[ ]:


row_r1 = a[1, :]    # 정수로 indexing한 경우 데이터의 차원이 떨어진다. 
row_r2 = a[1:2, :]  # slicing할 경우 차원이 유지된다.
row_r3 = a[[1], :]  # 리스트로 indexing할 경우 차원이 유지된다
print(row_r1, row_r1.shape) 
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


# 정수의 array로 indexing하기

# In[ ]:


a = np.array([[1,2], [3, 4], [5, 6]])
print(a)

print(a[[0, 1, 2], [0, 1, 0]])  #[0, 1, 2] : 1차원 index 지정 //  [0, 1, 0] : 2차원 index
print(np.array([a[0, 0], a[1, 1], a[2, 0]])) # 위의 표현은 아래와 같다


# In[ ]:


print(a[[0, 0], [1, 1]]) #같은 위치를 반복해서 indexing할 수도 있다
print(np.array([a[0, 1], a[0, 1]])) # 위의 표현은 아래와 같다


# 이를 이용하면 행렬의 각 행에서 서로 다른 위치의 요소에 접근할 수 있다.

# In[ ]:


a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# In[ ]:


b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])  # 각 행마다 b에 요소를 가져온다


# In[ ]:


a[np.arange(4), b] += 10   #값을 변경한다
print(a)


# Boolean array를 이용하여 indexing 하면 원하는 위치의 요소들을 자유롭게 가져올 수 있다.

# In[ ]:


import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])
print(a)

bool_idx = (a > 2)  # 2보다 큰 element를 찾아 boolean array를 반환한다.
print(bool_idx) 


# In[ ]:


print(a[bool_idx]) #True에 해당하는 요소들만 가져온다.
print(a[a > 2])    # 더 간단하게 표현할 수도 있다


# ### Datatypes

# In[ ]:


x = np.array([1, 2])      # numpy가 데이터 타입을 자동으로 추정한다.
y = np.array([1.0, 2.0]) # numpy가 데이터 타입을 자동으로 추정한다.
z = np.array([1, 2], dtype=np.int64)  # 데이터 타입을 지정한다.

print(x.dtype, y.dtype, z.dtype)


# 타입에 주의할것

# In[ ]:


x = np.zeros((2,2), dtype = np.int64)
x[1,1] = 0.1
print(x)


# ### Array math
# 
# 기본적인 행렬 또는 텐서연산은 operator overloading으로 구현되어 있으며 함수 이름을 이용하여 사용할 수도 있다.

# In[ ]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum
print(x + y)
print(np.add(x, y))


# In[ ]:


# Elementwise difference; 
print(x - y)
print(np.subtract(x, y))


# In[ ]:


# Elementwise product; 
print(x * y)
print(np.multiply(x, y))


# In[ ]:


# Elementwise division; 
print(x / y)
print(np.divide(x, y))


# In[ ]:


# Elementwise square root; produces the array
print(np.sqrt(x))


# `*` 이 elementwise multiplication임에 주의하라.
# 
# vector간의 내적이나, 행렬과 벡터의 곱, 혹은 행렬간의 곱은 ``dot``을 이용한다

# In[ ]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

#백터간의 내적
print(v.dot(w))     # 인스턴스 메서드를 이용해 접근
print(np.dot(v, w)) # 함수를 이용


# In[ ]:


# Matrix / vector product;
print(x.dot(v))
print(np.dot(x, v))


# In[ ]:


# Matrix / matrix product;
print(x.dot(y))
print(np.dot(x, y))


# matmul 함수를 이용할 수도 있다. 3차원 이상의 행렬일때는 실행결과가 다르다.

# In[ ]:


print(np.matmul(x, y))
print(x @ y)


# 그외에도 numpy에는 다양한 함수를 제공하며 자세한 내용은 [documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html)를 참고하기 바람
# 

# In[ ]:


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"


# ### Reshape
# 
# reshape함수를 이용하여 배열의 모양을 변경할 수 있다.

# In[ ]:


a = [1,2,3,4,5,6,7,8]
b = np.reshape(a,(2,4))
c = np.reshape(a,(4,2))

print(b)
print(c)


# ``T`` attribute를 이용하여 transpose 할 수 있다

# In[ ]:


print(x)
print(x.T)


# ### Broadcasting

# Broadcasting 서로 다른 모양의 array간 연산에 아주 유용하다.
# 
# 주로 우리는 크기가 큰 array와 작은 array가 있고, 크기가 작은 array를 반복해서 사용하고자 한다.
# 
# 예를들어 한 행렬의 각 행마다 constant vector를 더한다고 생각해보자

# In[ ]:


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(x)
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

for i in range(4):
    y[i, :] = x[i, :] + v

print(y)


# 이렇게 구현할 수도 있지만 파이썬으로 for루프를 순회하는것은 매우 느리다.
# 
# vector `v`를 행렬 `x`의 각 행에 더하는 것은 v를 세로로 쌓아 행렬 `vv` 만들어 더하는것과 같다.

# In[ ]:


vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                

y = x + vv  # Add x and vv elementwise
print(y)


# Numpy broadcasting 이를 자동으로 해준다.

# In[ ]:


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)


# Broadcasting은 간단하게는 다음의 규칙을 갖는다.
# 
# - 두 array의 차원이 다르면 차원이 낮은 array를 덧붙여 같은 차원으로 만든다.
# 
# 더 자세하게는 [documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) 과 [설명](http://wiki.scipy.org/EricsBroadcastingDoc)을 참고할것.
# 
# broadcasting을 지원하는 함수는 universal functions이라고 불린다. [documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)을 참고할것
# 
# 

# In[ ]:


x = np.random.random((3, 4))
y = np.random.random((3, 1))
z = np.random.random((1, 4))

print(x.shape)
print(y.shape)
print(s.shape)

s = x + y ## y is broadcasted along dim 1.
p = x * z ## z is broadcasted along dim 0.


# In[ ]:


v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# 외적을 계산하기 위해 먼저 v를 reshape하여 shape (3, 1)인 컬럼 벡터를 만든다.
# 그 후 broadcast 를 통해 (3, 2)인 output을 얻는다.

print(np.reshape(v, (3, 1)) * w)


# In[ ]:


# 행렬의 각 행에 벡터 더하기
x = np.array([[1,2,3], [4,5,6]])
# x는 shape (2, 3), v는 shape (3,) 이며 (2, 3)으로 broadcase된다

print(x + v)


# In[ ]:


# 행렬의 각 열에 벡터 다하기
# x는 shape (2, 3) and w는 shape (2,)이므로 transpose(x)를 통해 shape (3, 2)를 얻어 더한다.
print((x.T + w).T)


# In[ ]:


# 또다른 방법은 w를 reshape하여 shape (2, 1)로 만드는 것이다.
# 그러면 바로 broadcasting이 가능하다.
print(x + np.reshape(w, (2, 1)))


# In[ ]:


# 행렬의 scalar곱:
# Numpy는 scalar를 shape ()인 array로 생각하여, shape (2, 3)으로 broadcase하여 계산한다.
print(x * 2)


# 파이썬을 사용할때 explicit for-loops 는 최대한 피해야 한다.
# 코드 실행 속도가 ~10-100x 이상 느려진다.
# 
# Broadcasting은 코드를 간편하고 빠르게 만듦으로 가능한 한 자주 사용하려고 노력해야 한다.

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'x = np.random.rand(1000, 1000)\nfor i in range(100, 1000):\n    for j in range(x.shape[1]): \n        x[i, j] += 5\n')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'x = np.random.rand(1000, 1000)\nx[np.arange(100,1000), :] += 5 \n')


# ## Matplotlib
# 
# Matplotlib은 plot을 위한 라이브러리이며, 여기서는  `matplotlib.pyplot` 모듈에 대해 간략히 다룰 것이다.

# In[ ]:


import matplotlib.pyplot as plt


# ``plot``을 통해 이차원 데이터를 plot할수 있다

# In[ ]:


# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)


# 여러 라인을 플롯할수도 있고 title, legend, and axis 라벨을 붙일수 있다

# In[ ]:


y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])


# ### Subplots 
# 다양한 plot들을 ``subplot``을 이용하여 한 그림에 그릴수 있다

# In[ ]:


# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

