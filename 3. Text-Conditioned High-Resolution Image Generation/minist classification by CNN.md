---
jupyter:
  jupytext:
    notebook_metadata_filter: nbsphinx
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  nbsphinx:
    execute: never
---

# 실습 9: CNN을 이용한 MNIST 분류


본 실습에서 사용하는 데이터셋(MNIST)의 출처는 다음과 같음을 밝힌다.

Raw Data 출처: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition.", Proceedings of the IEEE, 86(11):2278-2324, November 1998.

편집자1: Yann LeCun, Corinna Cortes, Christopher JC Burges, 《THE MNIST DATABASE of handwritten digits》, 〈http://yann.lecun.com/exdb/mnist/〉, 방문 날짜 2022.03.01.

편집자2: MIT Beaverworks에서 편집자1의 웹사이트의 데이터셋을 쉽게 다운 받을 수 있는 라이브러리 datasets를 제작


이번 실습에서는 유명한 데이터셋인 MNIST를 이용하여 CNN을 학습시킬 것이다. MNIST 데이터셋은 0부터 9까지의 숫자를 손으로 쓴 이미지와 그에 대응되는 숫자 라벨로 이루어져 있는 데이터셋이다. 학습용 데이터셋은 60,000개의 이미지로, 테스트용 데이터셋은 10,000개의 이미지로 이루어져 있다. 따라서 MNIST 데이터셋을 이용하여 CNN을 학습시키면, 0부터 9까지의 숫자를 손으로 쓴 글씨 이미지들을 분류하는 신경망 모델을 얻을 수 있다.


이번 실습에서 필요한 기본적인 라이브러리들을 먼저 import 하자.

```python
import numpy as np
import mygrad as mg
from mygrad import Tensor

import matplotlib.pyplot as plt

%matplotlib notebook
```


### Step 1. MNIST 데이터 관찰하기


셋업을 잘 진행했다면, 현재 주피터 노트북 파일이 들어있는 폴더의 상위폴더에 Datasets 폴더가 이미 있을 것이다. 먼저 mnist.npz 데이터셋이 Datasets 폴더에 잘 다운로드되어 있는지 확인해보자. 없다면 첫번째 실습이었던 셋업이 잘 이루어지지 않은 것이므로 다시 다운받아야 한다.

MNIST 데이터셋을 다운로드하는 다른 방법으로는 아래 코드를 이용하여 데이터셋을 다운로드하는 방법이 있다. 셋업 단계에서 패키지를 설치할 때 깃허브에서 datasets 패키지를 설치했었기 때문에, 아래 코드만으로 다운로드할 수도 있다. 이미 정확한 위치에 제대로 다운로드되어 있다면 아래 코드를 실행했을 때 File already exists: 라는 메시지가 출력될 것이다.

```python
from datasets import load_mnist, download_mnist
download_mnist()
```


MNIST 데이터셋을 구성하는 학습용 데이터셋과 테스트용 데이터셋을 가져오자.

```python
# mnist 데이터셋을 불러와서 학습용/테스트용 이미지/라벨 을 저장
x_train, y_train, x_test, y_test = load_mnist()
```


x_train, y_train, x_test, y_test는 모두 numpy.ndarray 이다. 각각의 shape과 데이터 타입(dtype)을 확인해보자. 또한, x_train과 x_test의 shape으로부터 데이터셋을 구성하는 이미지가 각각 몇 개의 색상 채널로 이루어져 있는지 확인하자.

```python
type(x_train), x_train.shape, x_train.dtype
```

```python
type(y_train), y_train.shape, y_train.dtype
```

```python
type(x_test), x_test.shape, x_test.dtype
```

```python
type(y_test), y_test.shape, y_test.dtype
```

이제 MNIST 데이터셋이 어떤 데이터로 구성되어 있는지 확인하기 위해 아래 코드를 사용해보자. matplotlib을 이용하여 x_train의 이미지를 띄우고, 제목으로 y_train의 truth 값을 출력할 것이다. img_id 값을 바꾸어가며 MNIST 데이터셋을 이루는 데이터들을 확인해보자.

```python
img_id = 5

fig, ax = plt.subplots()
ax.imshow(x_train[img_id, 0], cmap="gray")
ax.set_title(f"truth: {y_train[img_id]}");
```


### Step 2. 학습의 큰 틀 구상하기


신경망의 학습을 이용하여 문제를 해결하려 할 때 다음 질문들에 답해야 한다.

> 1. 우리가 해결하려고 하는 문제의 목적은 무엇인가?
>
> 2. 이 문제를 해결하기 위해 어떤 데이터셋이 필요한가? 어떻게 마련해야 할까?
>
> 3. 데이터를 신경망에 넣어주기 위해 어떤 전처리가 필요할까?
>
> 4. 이 문제를 해결하기 위해 어떤 신경망을 쓰는 게 좋을까? 각 계층을 어떻게 구성해야 할까?
>
> 5. 어떤 손실함수를 선택해야 할까? 정확도함수는 어떻게 정의할 수 있을까?
>
> 6. 어떤 Optimizer를 선택해야 할까?


하나씩 대답하면서 학습의 큰 틀을 구상해보자. (1번, 2번 질문에 대해서는 이미 대답을 알고있지만, 새로운 문제에 직면했다고 생각하고 따라가 보자.)


**Q1.** 우리가 해결하려고 하는 문제의 목적은 무엇인가?

A1. 0부터 9까지의 숫자를 손으로 쓴 글씨 이미지들을 분류하는 문제를 해결하고자 한다.


**Q2.** 이 문제를 해결하기 위해 어떤 데이터셋이 필요한가? 어떻게 마련해야 할까?

A2. Step 1에서 살펴본 MNIST 데이터셋은 우리가 해결하려는 문제에 적합한 데이터셋이다. 따라서 이를 학습에 이용하면 될 것이다. 만약 이러한 데이터셋이 세상에 존재하지 않거나 공개되어 있지 않았다면, 데이터셋을 직접 만드는 데 오랜 시간이 걸렸을 것이다.


**Q3.** 데이터를 신경망에 넣어주기 위해 어떤 전처리가 필요할까?

A3. 그동안의 실습에서 진행했던 전처리와 유사하게, 데이터 타입을 바꾸는 작업과 데이터에 대한 특성 스케일링(정규화)을 진행하면 된다. 그리고 5절 이론에서 배운 패딩(padding)을 적용해주는 전처리도 추가적으로 필요하다.


**Q4.** 이 문제를 해결하기 위해 어떤 신경망을 쓰는 게 좋을까? 각 계층을 어떻게 구성해야 할까?

A4. 이미지에 관한 학습이므로 합성곱 계층과 최대 풀링 계층을 사용한 CNN을 사용하는 것이 좋다. 우리는 합성곱 계층과 풀링 계층으로 구성된 은닉층 두 개와 밀집층으로 구성된 은닉층 하나, 밀집층으로 구성된 출력층 한 개로 이루어진 4층 CNN을 사용해볼 것이다. 활성함수는 모든 은닉층에서 ReLU이고, 출력층에서는 Softmax 함수이다.

```
CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FLATTEN -> DENSE -> RELU -> DENSE -> SOFTMAX
```

우리는 분류 문제를 해결해야 하므로, 입력 데이터가 각 분류 항목에 해당할 확률을 항목별 인덱스에 대응되는 배열로 얻기를 바란다. 즉, 입력된 이미지가 숫자 $i$일 확률을 $p_i$라 할 때 출력층에서는 $[p_0, p_1, p_2, \cdots, p_9]$가 출력되도록 할 것이다. 따라서 각 뉴런의 출력값이 0과 1 사이의 값을 갖고 합은 1이 되는 조건을 만족할 수 있도록, 출력층의 활성함수를 Softmax 함수로 선택하는 것이 좋은 선택임을 알 수 있다.


**Q5.** 어떤 손실함수를 선택해야 할까? 정확도함수는 어떻게 정의할 수 있을까?

A5. 분류 문제에서 유용하게 쓰이는 교차 엔트로피 함수를 손실함수로 사용하자. 그리고 성능을 평가하기 위한 지표로 손실(loss)과 더불어 정확도(accuracy)를 사용할 것이다. 분류 문제이므로 정확도는 맞으면 1, 틀리면 0으로 하여 평균을 계산하면 될 것이다.


**Q6.** 어떤 Optimizer를 선택해야 할까?

A6. 경사하강법 다음으로 기본적인 Optimizer인 SGD를 사용하자.


### Step 3. 데이터 전처리


진행해야 하는 데이터 전처리는 크게 세가지이다.

1) 데이터 타입을 신경망에서 사용할 수 있도록 float으로 바꾸어준다.

2) 데이터에 대한 특성 스케일링(정규화)을 진행한다.

3) 첫번째 합성곱 연산 후에 이미지의 shape이 유지되도록 패딩을 사용한다.


먼저, 원본 이미지에는 각 픽셀에 해당하는 값으로 부호 없는 8비트 정수(uint8)가 저장되어있다. 그러나 우리는 신경망 각 계층의 연산을 수행하기 위해 이를 float형으로 변환해야 한다. .astype()을 사용하여 배열 x_train과 x_test의 데이터 타입을 np.float32로 바꾸어주자. (라벨인 y_train과 y_test의 데이터 타입은 상관이 없다.)

```python
# <COGINST>
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
# </COGINST>
```


두번째로, 우리는 학습이 잘 진행되도록 하기 위해 이미지 데이터들의 특성 스케일링(정규화)을 진행해줄 것이다. 각 픽셀 값은 [0, 255]의 범위로 경계가 분명하게 고정되어 있기 때문에, 최소-최대 정규화를 사용하기 적절하다. 그런데 이 실습에서는 최솟값과 최댓값이 각각 0(검은색)과 255(흰색)이다. 따라서 모든 픽셀 값을 255로 나누어주기만 하면 최소-최대 정규화를 통해 그 값이 [0, 1]의 범위로 바뀌도록 구현할 수 있다.

```python
# <COGINST>
x_train /=  255.
x_test /= 255.

print(x_test.shape)
# </COGINST>
```


마지막으로, 이미지 둘레에 패딩을 도입해보자. shape 5x5인 필터를 스트라이드 1로 이용할 것이므로, 첫번째 합성곱 연산 후에 이미지의 shape이 유지되도록 하기 위해서는 이미지의 모든 면에 0으로 이루어진 두 개의 행/열을 패딩해주면 된다. 우리가 갖고 있는 원래의 이미지가 28x28이므로, 패딩 후의 이미지는 32x32가 될 것이다. 

참고로, 패딩의 폭은 하이퍼파라미터로, 합성곱 계층을 정의할 때 그 값을 지정해주기도 한다. 하지만 이 실습에서는 학습 전 전처리 작업으로 패딩을 도입하였다.

```python
x_train = np.pad(x_train, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
x_test = np.pad(x_test, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
```


### Step 4. 모델(클래스) 정의하기


#### 가중치 초기화


우리에게 이제 클래스로 신경망 모델을 정의하는 것 쯤은 전혀 어렵지 않은 일이다. 바로 프로그래밍을 하고 싶겠지만, 그 전에 가중치를 초기화하는 방법에 관해 하나만 더 알아보자.

우리는 그동안 가중치를 초기화하기 위해 다양한 initializer (uniform, normal, he_normal)를 사용하였다. 이번에는 가중치를 초기화하기 위해 Xavier Glorot이 Yoshua Bengio와 함께 작성한 [논문](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 에서 제안한 **자비에 초기화(Xavier Initialization)** 방법 중, **glorot_uniform**을 initializer로 이용할 것이다. 편향은 0으로 초기화할 것이다.


glorot_uniform 초기화 방식은 다음의 균등분포를 따르도록 초기화하는 것이다. $N_{in}$과 $N_{out}$은 각각 들어오는 입력 수와 내보내는 출력 수이다. 핵심은 들어오는 쪽의 뉴런 수만 고려했던 He 초기화(He Initialization)와 달리 내보내는 쪽도 함께 고려한다는 점이다.


\begin{equation}
W \sim U(-\frac{\sqrt 6}{\sqrt{N_{in}+N_{out}}},+\frac{\sqrt 6}{\sqrt{N_{in}+N_{out}}})
\end{equation}


mygrad.nnet.initializers의 자비에 초기화 함수들(glorot_normal, glorot_uniform)과 He 초기화 함수들(he_normal, he_uniform)에는 모두 “gain”이라는 매개변수(parameter)가 정의되어 있다. gain은 가중치 배열에 곱해지는 스케일링 인자(상수배 해주는 역할. scaling factor)이며, 디폴트 값으로 1을 가진다. 이 실습에서는 gain을 $\sqrt{2}$로 사용할 것이다.


가중치 초기화 방법 (weight_initializer)은 밀집층 혹은 합성곱 계층을 초기화할 때 아래 코드와 같이 지정해주면 된다. 이때, weight_kwargs에 weight_initializer가 필요로 하는 모든 매개변수들을 딕셔너리 형태로 전달해줄 수 있다. 이 실습에서는 gain을 $\sqrt2$로 하는 것 외에 다른 매개변수는 필요로 하지 않으므로 gain만 전달해주면 된다.

<!-- #region -->
```python
from mygrad.nnet.initializers import glorot_uniform

gain = {'gain': np.sqrt(2)}

# conv layer, dense layer 등을 초기화할 때 다음과 같이 사용
dense(d1, d2, 
      weight_initializer=glorot_uniform, 
      weight_kwargs=gain)
```
<!-- #endregion -->

#### 모델(클래스) 정의하기


이제 위에서 구상했던 계층 정보대로 신경망 모델을 클래스로 작성하면 된다. 우리가 구성할 신경망은 아래와 같이 두 개의 합성곱-풀링 계층과 두 개의 밀집층으로 이루어진 4층 신경망이다.

계층을 어떻게 구성하면 좋을지 각 계층의 하이퍼파라미터들을 결정해보자. 학습이 괜찮게 이루어지는 각 계층의 정보는 다음과 같다. (하이퍼파라미터로 다양한 값들을 시도해보아도 좋다.)

> 계층1_CONV: 5x5필터 20개, 스트라이드-1
>
> 계층1_POOL: 2x2, (스트라이드-2)
>
> 계층2_CONV: 5x5필터 10개, 스트라이드-1
>
> 계층2_POOL: 2x2, (스트라이드-2)
>
> 계층3_DENSE: 뉴런 20개
>
> 계층4_DENSE: 뉴런 10개 (**주의**: $[p_0, p_1, p_2, \cdots, p_9]$ 가 출력되어야 한다.)


위 계층 정보 중 필터와 풀링 윈도우의 shape 및 스트라이드는 클래스를 정의하는 단계에서 사용해야 하는 정보이고, 나머지 정보인 필터의 개수와 뉴런의 개수는 클래스로부터 객체를 생성하여 모델을 초기화할 때 지정해줄 것이다.


MyNN의 conv(), dense() 함수를 사용하여 계층을 생성할 때 전달해주어야 하는 매개변수가 무엇인지 알아보자.

일단 dense() 함수는 앞선 다른 실습들에서도 살펴보았듯 input_size(입력 데이터 수)와 output_size(출력 수 = 뉴런 수)를 전달해주어야 하고, 위에서와 같이 glorot_uniform 초기화를 위해 weight_initializer와 weight_kwargs를 지정해줄 수 있다.

그리고 conv() 함수는 input_size(입력 채널 수)와 output_size(출력 채널 수), filter_dims(필터 size)를 전달해주어야 하고, dense()와 마찬가지 방법으로 glorot_uniform 초기화를 진행할 수 있다. 특히 filter_dims는 전달 가능한 개수가 정해지지 않은 \*args 형태이다. 이 실습의 경우 $5 \times 5$ 필터를 사용하므로, 5,5를 전달해주면 된다.

```python
from mynn.layers.conv import conv
from mynn.layers.dense import dense

from mygrad.nnet.initializers import glorot_uniform
from mygrad.nnet.activations import relu
from mygrad.nnet.layers import max_pool

class Model:
    ''' 간단한 CNN (Convolutional Neural Network) '''
    
    def __init__(self, num_input_channels, f1, f2, d1, num_classes):
        """
        클래스 생성 시 실행되어 계층과 가중치들을 초기화하는 역할.
        
        매개변수 (Parameters)
        ----------
        num_input_channels : int
            입력 데이터(이미지)의 색상 채널 수
            
        f1 : int
            첫번째 계층 _ 합성곱 계층의 필터 수
        
        f2 : int
            두번째 계층 _ 합성곱 계층의 필터 수

        d1 : int
            세번째 계층 _ 밀집층의 뉴런 수
        
        num_classes : int
            모델의 분류 항목 수
            네번째 계층 _ 밀집층의 뉴런 수
        """
        # 두 개의 합성곱 계층과 두 개의 밀집층을 각각 초기화
        # MyNN의 conv(), dense() 함수 사용
        # weight_initializer = glorot_uniform, gain = np.sqrt(2) 로 설정
        
        # 주의: 두 번째 계층의 계산 결과로 얻은 이미지 채널을 벡터화(vectorization, flatten)한 결과가
        # 세 번째 계층으로 들어가므로, 세 번째 계층의 밀집층의 input_size를 잘 계산해야 함
        
        # <COGINST>
        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(num_input_channels, f1, 5, 5, 
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.conv2 = conv(f1, f2, 5, 5 ,
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.dense1 = dense(f2 * 5 * 5, d1, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)
        self.dense2 = dense(d1, num_classes, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)
        # </COGINST>


    def __call__(self, x):
        '''
        입력 데이터에 따른 모델의 순전파(Forward Propagation)를 수행
        
        매개변수 (Parameters)
        ----------
        x : numpy.ndarray, shape=(N, 1, 32, 32)
            입력 데이터. N장의 이미지
            
        반환 값 (Returns)
        -------
        mygrad.Tensor, shape=(N, num_classes)
            N장의 이미지에 대해 분류 항목별 예측값을 배열로 반환
            Softmax 통과 전이므로 확률이 아님에 주의
        '''
        # 모델의 구조를 잘 생각하며 모델의 순전파를 수행하는 코드 작성
        # 가중치를 포함하는 합성곱 계층 및 밀집층은 __init__에서 정의함
        # 활성함수 relu와 최대 풀링 함수 max_pool을 추가로 이용하여 작성하면 됨
        
        # 이미지 채널을 밀집층에 넣어주기 위한 벡터화 진행 시,
        # N개의 이미지에 대해 개별적으로 벡터화를 진행해야 함 (HINT > NumPy ndarray의 reshape() 메서드 사용) 
        
        # <COGINST>
        x = relu(self.conv1(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.conv2(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)
        # </COGINST>
        

    @property
    def parameters(self):
        """
        모델 파라미터를 리스트 형태로 전부 가져올 수 있는 유용한 함수
        데코레이터 @property를 붙였기에 메서드가 아닌 속성처럼 사용해야 함.
        즉, model.parameters()가 아닌 model.parameters로 호출
        
        반환 값 (Returns)
        -------
        List[Tensor, ...]
            모델의 학습 가능한 모델 파라미터들을 모아놓은 리스트
        """
        # __init__에서 정의한 4개의 계층에 포함된 모든 가중치(모델 파라미터) 반환
        # 각 레이어의 파라미터를 self.conv.parameters와 같이 가져올 수 있음
        # for문 이용하기
        
        # <COGINST>
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params
        # </COGINST>

```

### Step 5. 정확도 함수


분류의 정확도를 출력하는 함수를 작성하자. 정확도는 맞으면 1, 틀리면 0으로 하여 평균을 계산하면 된다.

```python
def accuracy(predictions, truth):
    """
    하나의 batch에 대한 모델의 예측값과 실제값을 비교하여, 정확도를 계산하는 함수
    
    매개변수 (Parameters)
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        D개의 분류 항목에 대해 예측한 각 확률을 batch의 데이터 M개에 대해 배열로 정리
        이 실습에서는 분류할 항목의 개수가 숫자 0~9, 총 10개이므로 D = 10
        
    truth : numpy.ndarray, shape=(M,)
        데이터 M개에 대응되는 정답 라벨들로 이루어진 배열
        각 라벨은 D개의 분류 항목 [0, D) 중 하나의 값
            
    반환 값 (Returns)
    -------
    float
        올바른 예측의 비율 (0 이상 1 이하의 실수값)
        해당 batch에 대한 모델의 분류 정확도
    """
    return np.mean(np.argmax(predictions, axis=1) == truth) # <COGLINE>
```

### Step 6. 학습시키기


앞서 구상 단계에서 우리는 손실함수는 교차 엔트로피 함수, Optimizer는 MyNN의 SGD로 선택하기로 하였다. 학습을 시키는 단계 또한 앞서 여러 차례 프로그래밍 해봤던 부분이므로 잘 할 수 있을 것이다.


#### 모델 및 Optimizer 초기화


먼저 Optimizer를 MyNN의 SGD로 선택하여 모델을 초기화하자. Optimizer의 파라미터, 학습률(learning_rate) 외에 가중치 감소(weight_decay)와 모멘텀(momentum)이라는 값을 추가할 것이다.

먼저 가중치 감소는 과적합(overfitting)을 방지하기 위한 방법 중 하나이다. 가중치의 값이 크면 과적합이 발생하는 경우가 많기 때문에, 가중치 값이 클수록 큰 페널티를 부과함으로써 가중치가 커지는 것을 방해한다. 이번 실습에서는 가중치 감소의 정도를 결정하는 하이퍼파라미터 값을 $5\times10^{-4}$로 설정하면 적당하다.

두 번째로 모멘텀은 물리에서의 ‘관성’을 가중치 업데이트에 반영한 것이다. 가중치가 기존에 움직이던 속도에 비례하여 해당 방향으로 계속 이동하도록 하는 힘이 작용한다고 이해하면 된다. 이를 통해 편미분 계수가 급격히 달라지더라도 좀 더 부드럽게 움직이도록 할 수 있다. 이번 실습에서는 기존에 움직이던 정도를 얼마나 반영할 것인지에 관한 하이퍼파라미터 값을 0.9로 설정하면 적당하다.

```python
# SGD를 import하여
# Optimizer로 모델을 초기화

from mynn.optimizers.sgd import SGD

model = Model(f1=20, f2=10, d1=20, num_input_channels=1, num_classes=10)
optim = SGD(model.parameters, learning_rate=0.01, momentum=0.9, weight_decay=5e-04)
```

#### 손실 및 정확도의 변화를 실시간으로 나타낼 그래프 생성


앞선 다른 실습에서 했던 바와 같이, 학습이 진행됨에 따라 동적인 그래프로 손실과 정확도를 확인할 수 있도록 아래의 코드를 실행하자.

```python
from noggin import create_plot

plotter, fig, ax = create_plot(["loss", "accuracy"])
```

#### 학습시키기


batch size가 100인 batch들을 랜덤으로 구성하여, 2 epoch만큼 학습을 진행해보자. 이제껏 진행했던 학습 중 가장 데이터 수가 많고 깊이도 깊기 때문에 2 epoch만 진행함에도 시간이 꽤 걸릴 것이다. 그리고 더 좋은 학습 결과를 얻으려면 2 epoch로는 부족하고, 더 많은 학습이 필요할 것이라 짐작해볼 수 있다.

학습용 데이터셋으로부터 랜덤 batch를 생성하여 학습을 진행하는 코드는 이전에 여러 번 작성해본 바 있으므로 잘 작성할 수 있을 것이다. 여기서 한 가지 추가해볼 부분은 학습용 데이터셋에 대한 1 epoch가 끝날 때마다 테스트용 데이터셋에 대해 예측을 진행하고 테스트 정확도를 판단하는 부분이다. 이 과정은 모델이 처음 접한 데이터에 대해서도 잘 작동하는지 확인하기 위한 중요한 과정이다.

테스트용 데이터셋에 대해서도 batch를 나누어 학습의 정확도를 구한다. 이 부분에서는 가중치의 업데이트를 위한 역전파가 이루어지지 않는다. 또한 테스트용 데이터셋에 대한 정확도도 plotter로 나타내볼 수 있는데, 이때 테스트용 데이터셋에 대한 plotter.set_test_batch() 함수는 추적한 정확도를 일일이 그래프로 나타내지 않는다. 대신 잘 저장하고 있다가 plotter.set_test_epoch()에서 1 epoch 전체에 대한 평균 정확도를 구하여 그래프에 표시해준다. plotter.set_train_epoch()과 함께 진행함으로써 값을 비교할 수도 있다.

```python
from mygrad.nnet.losses import softmax_crossentropy

batch_size = 100

for epoch_cnt in range(2):
    
    # x_train 데이터셋의 랜덤 batch 구성하기 위해
    # 랜덤으로 섞인 idxs 배열 구성하기
    
    # <COGINST>
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)  
    # </COGINST>
    
    
    # 학습용 데이터셋의 batch들에 대해 학습을 진행 (앞선 실습들과 매우 유사함)
    # 주의 1: 손실함수로 softmax_crossentropy() 이용
    # 주의 2: 경사하강법을 진행할 때에는 Optimizer optim의 step() 실행
    # 주의 3: 학습 중 loss와 accuracy를 추적
    
    for batch_cnt in range(len(x_train)//batch_size):
        
        # <COGINST>
        # idxs 배열을 이용하여 batch_cnt번째 batch 구성하기
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices]  # random batch of our training data

        # prediction : batch에 대한 예측값
        prediction = model(batch)
        
        # truth : batch에 대한 실제값
        truth = y_train[batch_indices]
        
        # loss : softmax_crossentropy() 이용하여 loss 계산
        loss = softmax_crossentropy(prediction, truth)
        
        # 경사하강법 진행에 필요한 모든 편미분계수를 얻기 위해
        # loss에 대해 .backward() 메서드 실행
        loss.backward()

        # optim의 step() 실행하여 경사하강법 진행
        optim.step()
        
        # accuracy: 앞서 작성한 accuracy() 이용한 정확도 계산
        acc = accuracy(prediction, truth)
        
        # 학습 중 loss와 accuracy를 추적
        plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                 batch_size=batch_size)
        # </COGINST>
    
    
    # 테스트용 데이터셋의 batch들에 대해 모델을 평가
    
    for batch_cnt in range(0, len(x_test)//batch_size):
        
        # 학습 시와 마찬가지로 같은 batch size의 랜덤 batch를 구성
        
        # <COGINST>
        idxs = np.arange(len(x_test))
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_test[batch_indices] 
        # </COGINST>
        
        
        # 역전파가 필요없는 부분이므로, 그래디언트 계산이 이뤄지지 않도록
        with mg.no_autodiff:
            # prediction : 테스트 batch에 대한 예측값
            prediction = model(batch)
            
            # truth : 테스트 batch에 대한 실제값
            truth = y_test[batch_indices]
            
            # accuracy: 앞서 작성한 accuracy() 이용한 정확도 계산
            acc = accuracy(prediction, truth)
        
        # noggin으로 test-accuracy 추가
        plotter.set_test_batch({"accuracy": acc}, batch_size=batch_size)
    
    plotter.set_train_epoch()
    plotter.set_test_epoch()
plotter.plot()
```


### Step 7. 학습 결과 확인하기


학습 결과를 확인하기 위해, 테스트용 데이터셋에 대해 이미지와 그에 대응되는 모델의 예측 결과를 시각화해보자.

```python
_, _, img_test, label_test = load_mnist()

# MNIST의 라벨로 이루어진 튜플.
# ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
labels = load_mnist.labels  
```

plot_model_prediction() 함수를 작성하여, index만 입력해주면 이미지와 실제값, 예측값까지 한 번에 손쉽게 시각화할 수 있도록 하자. 실습 앞부분에서 MNIST 이미지를 시각화하는 함수를 이미 살펴본 바 있으니 참고하자.

```python
def plot_model_prediction(index):
    '''index를 입력받아 index에 해당하는 이미지와 예측 결과를 시각화하는 함수'''

    true_label_index = label_test[index]
    true_label = load_mnist.labels[true_label_index]

    with mg.no_autodiff:
        # model에 입력해주는 데이터는 4차원이어야 하기 때문에
        # 축 하나가 사라지지 않도록 인덱스 설정
        prediction = model(x_test[index : index + 1])

        # 가장 점수가 높은 라벨이 예측값이 됨
        predicted_label_index = np.argmax(prediction.data, axis=1).item()
        predicted_label = labels[predicted_label_index]
    
    
    # 이미지, 실제 라벨, 예측 라벨 모두 시각화
    
    # <COGINST>
    fig, ax = plt.subplots()
    ax.imshow(img_test[index, 0], cmap="gray")
    
    # shape-(H, W, C), 픽셀값 데이터 타입 uint8 이어야 한다.
    # img = img_test[index].transpose(1, 2, 0).astype("uint8")
    # ax.imshow(img)
    
    ax.set_title(f"Predicted: {predicted_label}\nTruth: {true_label}")
    # </COGINST>
    
    
    return fig, ax
```

```python
# x_test의 랜덤 데이터를 시각화
index = np.random.randint(0, len(x_test))
plot_model_prediction(index);
```

그다음, 실제 정답 라벨을 사용하여 모델이 잘못 예측한 사례들을 찾고, 그 중 일부를 그래프로 표시해보자. 이때도 plot_model_prediction() 함수를 사용하면 된다.

```python
# 예측이 틀린 모든 사례 찾아서 해당 인덱스를 bad_indices에 저장
bad_indices = []

for batch_cnt in range(0, len(x_test) // batch_size):
    idxs = np.arange(len(x_test))
    batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]
    batch = x_test[batch_indices]

    with mg.no_autodiff:
        # test-batch에 대한 모델의 예측 라벨
        prediction = np.argmax(model(batch), axis=1)

        # test-batch에 대한 실제 라벨
        truth = y_test[batch_indices]
        
        # 예측과 실제가 다른 경우의 모든 인덱스를 저장
        (bad,) = np.where(prediction != truth)
        if bad.size:
            bad_indices.extend(batch_indices[bad].tolist())
```

```python
# bad_indices의 랜덤 데이터를 시각화
index = np.random.randint(0, len(bad_indices))
plot_model_prediction(bad_indices[index]);
```

### 배운 내용 되돌아보기


이번 실습에서는 0부터 9까지의 손글씨 숫자 이미지를 분류할 수 있는 신경망 모델을 만들기 위해 MNIST 데이터셋을 이용하여 CNN을 학습시켰다. 이 실습에서는 대표적인 구조의 CNN을 구성해보고, 학습시킴으로써 CNN에 대해 잘 이해할 수 있었다. 그리고 그동안 배웠던 신경망 학습의 과정과 유사한 과정이 많아 복습이 많이 되었을 것이다. 배운 내용을 정리해보자.

- 유명한 데이터셋 중 하나인 MNIST 데이터셋이 어떻게 구성되어 있는지 탐색해보았다. 특히 shape을 통해 이미지가 몇 개의 색상 채널로 이루어져 있는지 확인하고, 데이터 타입을 통해 각 픽셀에 저장된 값이 uint8임을 확인하였다.

- 데이터 전처리를 크게 세 가지 진행하였다. 데이터 타입을 np.float32로 바꾸고, 특성 스케일링(최소-최대 정규화)을 진행하고, 이미지 둘레에 패딩을 도입해주었다.

- 패딩을 도입할 때에는 합성곱 연산 이후 원래 이미지의 shape을 유지할 수 있도록하는 폭을 결정해주었다. 패딩의 폭은 하이퍼파라미터로, 합성곱 계층을 정의할 때 지정해주기도 하는 값이다.

- 모델(클래스)를 정의할 때에는 가중치의 초기화 방법과 계층의 구성을 결정해야 한다.

- 가중치 초기화 방법으로는 자비에 초기화(Xavier Initialization) 중 하나인 glorot_uniform을 사용하였다.

- 계층을 구성할 때에는 합성곱-풀링 계층과 밀집층을 각각 두 개씩 총 4층 신경망으로 구성하였으며, 활성함수는 은닉층은 전부 ReLU를, 출력층은 분류문제를 해결하기 위해 Softmax 함수를 사용하였다.

- 이번 실습은 실습 6,7과 같은 분류 문제였기 때문에, 정확도 함수는 실습 6에서 살펴보았던 코드와 동일하게 작성해도 충분했다.

- Optimizer를 SGD로 선택하여 초기화하는 과정에서 ‘가중치 감소’와 ‘모멘텀’이라는 개념을 배웠다.

- 학습을 시키는 과정은 랜덤 batch를 구성하고 Optimizer로 학습시키는 그동안의 실습들과 크게 다르지 않았다. 다만, 데이터셋이 무척 커졌기 때문에 epoch 수를 2로 대폭 줄여 학습을 진행했다.

- 이번 실습에서는 학습용 데이터셋에 대한 1 epoch의 학습이 끝날 때마다 테스트용 데이터셋에 대한 정확도를 함께 평가하였다. 처음 접하는 데이터에 대해서도 분류가 잘 이루어지는지 평가하기 위함이었다.

- 학습을 진행하면서 noggin 라이브러리를 이용하여 손실과 정확도를 실시간으로 나타내었다. 그리고 1 epoch의 학습이 끝나면 학습용 데이터셋과 테스트용 데이터셋에 대한 정확도를 점으로 찍어 시각화함으로써 비교해볼 수 있었다.

- 학습이 모두 끝난 후, 테스트용 데이터셋에 대해 모델의 예측 결과를 시각화하여 관찰하였다. 잘못 분류한 이미지들이 많지 않았기 때문에, 잘못 분류한 이미지만 모아서 시각화해보기도 하였다.
