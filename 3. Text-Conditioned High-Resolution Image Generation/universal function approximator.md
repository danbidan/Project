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

# Universal Function Approximator 학습시키기


Universal Approximation Theorem에 의거하여 보편적인 함수 $f$의 근사함수 $F$의 모델을 $ F(\{v_i\}, \{\textbf{w}_i\}, \{b_i\}; \textbf{x}) = \sum_{i=1}^{N} v_{i}\varphi(\textbf{x} \cdot \textbf{w}_{i} + b_{i})$ 로 세우고 학습을 진행해볼 것이다.

학습용 데이터셋을 $\textbf{x}$와 $\textbf{x}$에 대응되는 truth 데이터 $f(\textbf{x})$의 데이터 쌍들로 구성한 후, 경사하강법을 이용하여 적절한 모델 파라미터 $\{v_i\}, \{w_i\}, \{b_i\}$ 를 구함으로써 보편적인 수학적 함수 $f$의 근사함수를 학습시킬 수 있다.


### 목표


이번 실습에서는 일변수 함수인 $f(x)=cos(x), x \in [-2\pi, 2\pi]$의 근사함수를 구해볼 것이다. 여기서 $f$가 일변수 스칼라 함수이기 때문에 $x$, $w_i$를 벡터가 아닌 스칼라로 생각할 수 있다. 그러나 추후에 고차원 벡터(길이가 긴 1차원 배열)를 입력받는 모델에 대해서도 확장하기 위해, 프로그래밍할 때 스칼라가 아닌 벡터(즉, 길이가 1인 1차원 배열)라고 생각하고 프로그래밍 해보자.

주의) 여기서 세 종류의 모델 파라미터 각각의 개수이자 뉴런의 개수인 $N$은 하이퍼파라미터에 속한다. 즉, 학습을 통해 컴퓨터가 직접 적절한 값을 찾아가는 것이 아니라, 인간이 직접 시행착오를 통해 적절히 찾아 지정해주어야 한다.


그동안의 실습에서 꾸준히 사용해온 대표적인 라이브러리들을 미리 import 한 후 본격적인 실습을 시작해보자.

```python
import matplotlib.pyplot as plt

import mygrad as mg
import numpy as np

%matplotlib notebook
```

### Step1. 모델(클래스) 정의하기 : 순전파


우리가 가장 먼저 할 일은 Universal Approximation Theorem에 따라 구하게 될 근사함수 $F(x)$의 형태를 모델(클래스)로 정의해주는 것이다. 선형모델에 비해 구현하기 까다롭지만, MyGrad와 NumPy 라이브러리의 벡터화된 연산을 이용하면 짧고 간단한 코드를 작성할 수 있다.


#### batch의 개념 짚고 넘어가기

<!-- #region -->
우리는 입력 데이터를 $M$개씩 묶어 batch $\{x_{j}\}_{j=1}^{M}$로 구성한 후, batch를 이루는 개별적인 데이터에 대해 독립적으로 순전파를 진행할 것이다. 즉, $M$개의 데이터를 모델에 한 번에 전달하여 대응되는 $M$개의 예측을 한 번에 산출할 것이다.

batch 내 개별 데이터에 대한 함숫값 $ F(\{v_i\}, \{{w}_i\}, \{b_i\}; x_j)$를 구할 때 사용되는 연산은 batch 내의 다른 데이터들에는 영향을 받지 않는 독립적인 연산임을 꼭 명심하자. 각 예측은 대응되는 입력 데이터에 대해서만 이루어지기 때문에 다음의 값들에만 의존한다.

- 모델 파라미터: $\{v_i\}, \{w_i\}, \{b_i\}$ 
- 데이터 $x_j$


하나의 batch 당 $M$개의 독립적인 예측을 진행해야 한다고 하면 흔히 for문 등의 반복문을 떠올릴 것이다. 그러나 batch를 이루는 입력 데이터 $M$개를 shape-$(M, 1)$인 NumPy 배열로 정의하면 NumPy의 벡터화된 연산을 사용하여 순전파를 한 번에 진행할 수 있다.
<!-- #endregion -->

#### 근사함수 수식을 코딩으로 구현하기


이제 근사함수 $ F(\{v_i\}, \{{w}_i\}, \{b_i\}; x) = \sum_{i=1}^{N} v_{i}\varphi(x \cdot {w}_{i} + b_{i})$ 의 식을 batch size가 $M$인 batch에 대해 어떻게 코딩할지 고민해보자.

먼저, $x \cdot w_{i}$는 MyGrad 라이브러리의 matmul을 이용하여 계산할 수 있다. 이 문제에서 $x$와 $w_i$는 1차원 벡터이다. 따라서 $x \cdot w_i$는 벡터 간의 내적을 뜻하지만, 동시에 행벡터와 열벡터의 행렬곱으로도 생각할 수 있기 때문에 matmul 연산을 사용할 수 있다.

행렬곱을 이용하여 $N$개의 $w_i$에 대해 $x \cdot w_{i}$를 한번에 계산해보자. 1차원 행벡터 $x$를 1차원 열벡터 $w_i$ $N$개를 열에 대해 늘어놓은 행렬 $W$와 행렬곱하면 다음과 같다.

\begin{equation}
\begin{pmatrix}x\end{pmatrix}
\begin{pmatrix}w_{1} & w_{2} & \cdots & w_N \end{pmatrix} = 
\begin{pmatrix}x \cdot w_{1} & x \cdot w_{2} & \cdots & x \cdot w_N \end{pmatrix}
\end{equation}

결과적으로, $N$개의 $x \cdot w_{i}$ 값이 담긴 shape-$(1, N)$인 행렬을 얻게 된다.


이제 batch를 이루는 $M$개의 입력 데이터 $x_j$에 대해 위의 행렬곱을 한 번에 계산해보자. 1차원 행벡터 $x_j$를 행에 대해 늘어놓은 행렬 $X$를 1차원 열벡터 $w_i$ N개를 열에 대해 늘어놓은 행렬 $W$와 행렬곱하면 다음과 같다.

\begin{equation}
\begin{pmatrix}x_1\\
x_2\\
\vdots\\
x_M
\end{pmatrix}
\begin{pmatrix}w_{1} & w_{2} & \cdots & w_N \end{pmatrix} = 
\begin{pmatrix}x_1 \cdot w_{1} & x_1 \cdot w_{2} & \cdots & x_1 \cdot w_N \\ 
x_2 \cdot w_{1} & x_2 \cdot w_{2} & \cdots & x_2 \cdot w_N \\
\vdots & \vdots & \ddots & \vdots \\
x_M \cdot w_{1} & x_M \cdot w_{2} & \cdots & x_M \cdot w_N 
\end{pmatrix}
\end{equation}

결과적으로, $MN$개의 $x_{j} \cdot w_{i}$ 값이 담긴 shape-$(M, N)$인 행렬을 얻게 된다.

<!-- #region -->
결국, $X$에 shape-$(M,1)$, $W$에 shape-$(1,N)$의 적절한 모양의 배열만 넣어주면 한 번의 행렬곱(mg.matmul) 계산으로 원하는 결과를 얻을 수 있음을 알 수 있다. 따라서 $F(\{v_i\}, \{w_i\}, \{b_i\}; x_j ) = \sum_{i=1}^{N} v_{i}\varphi(x_{j} \cdot w_{i} + b_{i})$ 에서 $x_{j} \cdot w_{i}$ 부분을 구하는 코드는 다음과 같다.

```python
mg.matmul(x, w)
```
<!-- #endregion -->

<!-- #region -->
그 다음으로는 브로드캐스팅 성질을 이용한다. 우리는 스칼라 $b_i$가 shape-$(M, N)$인 행렬의 i번째 열 전체에 각각 더해지길 바란다.

\begin{pmatrix}x_1 \cdot w_{1}+b_1 & x_1 \cdot w_{2}+b_2 & \cdots & x_1 \cdot w_N+b_N \\ 
x_2 \cdot w_{1}+b_1 & x_2 \cdot w_{2}+b_2 & \cdots & x_2 \cdot w_N+b_N \\
\vdots & \vdots & \ddots & \vdots \\
x_M \cdot w_{1}+b_1 & x_M \cdot w_{2}+b_2 & \cdots & x_M \cdot w_N+b_N 
\end{pmatrix}

$\{b_i\}$를 shape-$(1,N)$인 행렬로 정의해주면, NumPy와 MyGrad의 브로드캐스팅 성질에 의해 다음의 코드 한 줄로 해결이 된다. 즉, $x_{j} \cdot w_{i} + b_{i}$ 부분을 구하는 코드는 다음과 같다.

```python
mg.matmul(x, w)+b
```
<!-- #endregion -->

<!-- #region -->
그 다음, 우리는 행렬의 각 원소에 대한 시그모이드 함숫값을 구해야 한다.

\begin{pmatrix}\varphi(x_1 \cdot w_{1}+b_1) & \varphi(x_1 \cdot w_{2}+b_2) & \cdots & \varphi(x_1 \cdot w_N+b_N) \\ 
\varphi(x_2 \cdot w_{1}+b_1) & \varphi(x_2 \cdot w_{2}+b_2) & \cdots & \varphi(x_2 \cdot w_N+b_N) \\
\vdots & \vdots & \ddots & \vdots \\
\varphi(x_M \cdot w_{1}+b_1) & \varphi(x_M \cdot w_{2}+b_2) & \cdots & \varphi(x_M \cdot w_N+b_N) 
\end{pmatrix}

NumPy와 MyGrad에서 수학 연산을 하는 함수에 NumPy 배열 혹은 MyGrad 텐서를 넣어주면 행렬의 모든 원소에 수학 연산을 각각 적용한 행렬을 얻을 수 있다. sigmoid 함수는 앞서 살펴본대로 mygrad.nnet.activations에서 import 할 수 있다. 따라서 $\varphi(x_{j} \cdot w_{i} + b_{i})$ 부분을 구하는 코드는 다음과 같다.

```python
sigmoid(mg.matmul(x, w)+b)
```
<!-- #endregion -->

<!-- #region -->
마지막으로 우리는 위에서 구한 행렬의 각 행 별로 $i$번째 원소에 $v_i$를 곱한 후 $(i=1,2, \cdots, N)$에 대해 다 더해주어야 한다. 그래서 최종적으로 다음과 같은 shape-$(M,1)$인 행렬을 구하기를 바란다.

\begin{equation}
\begin{pmatrix}\sum_{i=1}^{N} v_{i}\varphi(x_{1} \cdot w_{i} + b_{i}) \\ 
\sum_{i=1}^{N} v_{i}\varphi(x_{2} \cdot w_{i} + b_{i}) \\
\vdots \\
\sum_{i=1}^{N} v_{i}\varphi(x_{M} \cdot w_{i} + b_{i})
\end{pmatrix}
\end{equation}

이 행렬을 행렬곱 형태로 나타내면 다음과 같다. $\{v_i\}$는 shape-$(N,1)$인 배열로 정의해주어야 함을 확인할 수 있다.

\begin{equation}
\begin{pmatrix}\varphi(x_1 \cdot w_{1}+b_1) & \varphi(x_1 \cdot w_{2}+b_2) & \cdots & \varphi(x_1 \cdot w_N+b_N) \\ 
\varphi(x_2 \cdot w_{1}+b_1) & \varphi(x_2 \cdot w_{2}+b_2) & \cdots & \varphi(x_2 \cdot w_N+b_N) \\
\vdots & \vdots & \ddots & \vdots \\
\varphi(x_M \cdot w_{1}+b_1) & \varphi(x_M \cdot w_{2}+b_2) & \cdots & \varphi(x_M \cdot w_N+b_N) 
\end{pmatrix}
\begin{pmatrix}v_1\\
v_2\\
\vdots\\
v_N
\end{pmatrix}=
\begin{pmatrix}\sum_{i=1}^{N} v_{i}\varphi(x_{1} \cdot w_{i} + b_{i}) \\ 
\sum_{i=1}^{N} v_{i}\varphi(x_{2} \cdot w_{i} + b_{i}) \\
\vdots \\
\sum_{i=1}^{N} v_{i}\varphi(x_{M} \cdot w_{i} + b_{i})
\end{pmatrix}
\end{equation}

이로써 batch size가 $M$인 입력데이터 $\{x_{j}\}_{j=1}^{M}$에 대해 $F(\{v_i\}, \{w_i\}, \{b_i\}; x_j ) = \sum_{i=1}^{N} v_{i}\varphi(x_{j} \cdot w_{i} + b_{i})$을 한 번에 계산하는 벡터화 연산 코드를 찾아낼 수 있었다.

```python
out1 = sigmoid(mg.matmul(x, w) + b)  # shape-(M,N)
model_out = mg.matmul(out1, v)       # shape-(M,1)
```

model_out이 shape-$(M, 1)$인 텐서이므로, batch size가 $M$인 경우 대응되는 예측값이 총 $M$개의 독립적인 값으로 나타난다는 것을 확인할 수 있다.
<!-- #endregion -->

#### 모델(클래스) 정의하기


이제, 아래의 Model 클래스를 완성해보자. 모델 파라미터 $\{w_i\}$, $\{b_i\}$, $\{v_i\}$가 어떤 형태이고 왜 그런 형태여야 하는지 이해한 것을 바탕으로 initialize_params() 메서드를 작성할 수 있을 것이다. 그리고 model_out의 식을 어떻게 프로그래밍하는지 이해했다면, \_\_call\_\_() 메서드를 작성할 수 있을 것이다. 주석을 보며 스스로 작성해보자.
(load_parameters 메서드는 학습 과정에 필요한 메서드는 아니고, Step 5에서 학습 과정을 시각적으로 확인하는 데 사용하기 위해 정의해둔 메서드이다.)

```python
from mygrad.nnet.initializers import normal
from mygrad.nnet.activations import sigmoid

class Model:
    
    def initialize_params(self, num_neurons: int):
        """
        모델 파라미터 값인 'self.w', 'self.b', 'self.v'를 랜덤하게 초기화
        
        `mygrad.nnet.initializers.normal`함수를 사용하여 w, b, v 에 텐서 값을 초기화
        표준정규분포 ~N(0,1)을 따른다는 가정하에 초기화가 진행되도록
        
        self.w : shape-(1, N)
        self.b : shape-(N,)
        self.v : shape-(N, 1)
        
        매개변수 (Parameters)
        ----------
        num_neurons : int
            뉴런의 개수. 하이퍼파라미터
        """
        # 표준정규분포(standard normal distribution)를 따르는 텐서 값으로
        # `self.w`, `self.b`, `self.v` 초기화
        
        # <COGINST>
        self.w = normal(1, num_neurons)
        self.b = normal(num_neurons)
        self.v = normal(num_neurons, 1)
        # </COGINST>
        
        
    def __init__(self, num_neurons: int):
        """
        클래스 생성 시 실행되어 뉴런 개수 N을 입력받아 파라미터들을 초기화하는 역할.

        매개변수 (Parameters)
        ----------
        num_neurons : int
            뉴런의 개수
        """
        # self.N 초기화 : `num_neurons`
        self.N = num_neurons
        
        # `self.w`, `self.b`, self.v` 초기화 : `self.initialize_params()` 이용 
        self.initialize_params(num_neurons)
        
    
    def __call__(self, x):
        """
        입력 데이터에 따른 모델의 순전파(Forward Propagation)를 수행
        모델을 이용하여 `x`에 따른 예측값을 계산
        
        매개변수 (Parameters)
        ----------
        x : array_like(numpy.ndarray 혹은 mygrad.Tensor 등), shape-(M, 1)
            M개의 입력 데이터 (하나의 batch)
        
        반환 값 (Returns)
        -------
        prediction : mygrad.Tensor, shape-(M, 1)
            근사함수의 형식에 따라 구한 M개의 예측값을 담은 텐서
        """
        
        # <COGINST>
        out1 = sigmoid(x @ self.w + self.b)  # matmul[(M,1) w/ (1, N)] + (N,) --> (M, N)
        return out1 @ self.v # matmul[(M, N) w/ (N, 1)] --> (M, 1)
        # </COGINST>
    
    
    @property
    def parameters(self):
        """
        모델 파라미터를 튜플 형태로 전부 가져올 수 있는 유용한 함수
        데코레이터 @property를 붙였기에 메서드가 아닌 속성처럼 사용해야 함.
        즉, model.parameters()가 아닌 model.parameters로 호출
        
        반환 값 (Returns)
        -------
        Tuple[Tensor, ...]
            모델의 학습 가능한 모델 파라미터들을 모아놓은 튜플
        """
        
        return (self.w, self.b, self.v)  # <COGLINE>
    
    
    def load_parameters(self, w, b, v):
        self.w = w
        self.b = b
        self.v = v
```

### Step2. 손실함수 작성하기


모델을 정의했으므로 우리는 순전파를 통해 초기 근사함수 $F(x)$의 값을 얻을 수 있다. 초기 근사함수 $F(x)$는 임의로 지정한 모델 파라미터에 의해 정의된 함수이므로, 당연히 $f(x)=cos(x)$와는 차이가 클 것이다. 그러나 우리는 Universal Approximation Theorem에 의해, 근사함수 $F(\textbf x)$가 임의의 $\varepsilon > 0$에 대해 $ | F( \textbf{x} ) - f ( \textbf{x} ) | < \varepsilon $를 만족하도록 만드는 모델 파라미터 값들이 반드시 존재한다는 것을 알고 있다. 따라서, 손실함수를 $\mathscr{L}(x) = | F(x) - f(x) | = | F( \{v_i\}, \{w_i\}, \{b_i\}; x ) - \cos ( x ) |$ 로 정의하여, 손실함수 값이 0에 가까워지도록 학습하는 것이 자연스럽다.


그런데 우리는 batch size가 $M$인 batch를 모델에 한 번에 입력해줄 것이므로, $M$개의 손실함수 값 $\mathscr{L}(x_j)= | F( \{v_i\}, \{w_i\}, \{b_i\}; x_j ) - \cos ( x_j ) | \quad(j = 1, \dots, M)$을 한 번에 얻게된다. 우리는 이렇게 얻은 $M$개의 손실에 대해 평균을 구하여, 평균적인 손실을 학습에 사용할 것이다. batch 전체의 평균적인 손실을 학습에 사용하면 모델의 가중치를 더 연속적이고 부드럽게 이동시킬 수 있다.

$L(\{v_i\}, \{w_i\}, \{b_i\}; \{x_k\} ) = \frac{1}{M}\sum_{j=1}^{M} | F(\{v_i\}, \{w_i\}, \{b_i\}; x_j ) - \cos ( x_{j} ) |$


다음 코드의 주석을 잘 보고, 손실함수를 작성해보자. MyGrad의 수학 연산 함수들을 이용하여 단 한 줄의 코드로 완성해보자.

```python
def loss_func(pred, true):
    """
    매개변수 (Parameters)
    ----------
    pred : mygrad.Tensor, shape=(M,)
    true : mygrad.Tensor, shape=(M,)
    
    반환 값 (Returns)
    -------
    mygrad.Tensor, shape=()
        batch size가 M인 batch의 예측값(prediction)에 대한 평균 손실
    """
    return mg.mean(mg.abs(pred - true)) # <COGLINE>
```

### Step3. 경사하강법 함수 작성하기 : 역전파


모델 파라미터에 해당하는 텐서들과 학습률을 매개변수로 받아 경사하강법을 진행하는 함수를 작성해보자.

```python
def gradient_step(tensors, learning_rate):
    """
    경사하강법의 표준 공식에 따라 gradient-step을 실행
    
    매개변수 (Parameters)
    ----------
    tensors : Union[Tensor, Iterable[Tensors]]
        단일 텐서, 혹은 텐서로 이루어진 iterable(리스트, 튜플 등) 모두 가능
        만약 특정 tensor에 대한 `tensor.grad`가 `None`인 경우, 업데이트를 건너 뜀

    learning_rate : float
        매 gradient-step에서의 학습률. 양수

    참고
    -----
    함수에서 진행되는 모든 경사하강은 tensor 내에서 바로 반영되므로, 반환 값이 없음
    """
    # <COGINST>
    if isinstance(tensors, mg.Tensor):
        # Only one tensor was provided. Pack
        # it into a list so it can be accessed via
        # iteration
        tensors = [tensors]

    for t in tensors:
        if t.grad is not None:
            t.data -= learning_rate * t.grad
    # </COGINST>
```

### Step4. 학습시키기 : 순전파와 역전파의 반복


#### 데이터 준비하기


함수 $f(x)$의 근사함수를 구하는 상황이므로 데이터셋을 준비하는 것이 무척 간단하다. 학습용 데이터셋은 $(x_n, y_n^{(true)})$ 쌍으로 구성하면 되는데, 이 문제 상황에서는 $y_n^{(true)} = cos(x_n)$이다. 예를 들어 구간 $[-2\pi, 2\pi]$에서 근사함수를 구하고 싶다면, 구간 $[-2\pi, 2\pi]$를 균등하게 1000개 정도의 구간으로 나눈 배열을 입력되는 $x$데이터(train_data)로 사용하고, $cos$ 연산을 적용하는 함수를 true_f로 정의한 후, truth_data 배열을 true_f에 통과하여 얻어진 배열을 truth 데이터로 사용하면 될 것이다.

이때, 학습용 데이터를 이루는 $x_n$ 값들은 NumPy 배열이나 MyGrad의 상수 텐서로 정의하여 .backward() 메서드의 수행과정에서 불필요한 편미분계수가 계산되지 않도록 해야 한다.


#### 학습시키기


학습시키기에 앞서 train_data와 truth를 잘 정의해주어야 한다. 또한, Step 1에서 정의한 Model 클래스의 객체 model을 생성하여 모델 파라미터들이 초기화되도록 해야 한다.

주석을 잘 보고 데이터와 모델을 준비해보자.

```python
# shape-(1000,1)인 학습용 데이터셋 {x_n} 정의하기
# NumPy 배열이나 MyGrad의 상수 텐서로 정의해야 함
# 배열의 이름은 train_data로 지정하기

train_data = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(1000, 1) # <COGLINE>


# true_f(x) 정의하기
# x를 입력받아 np.cos(x)를 반환하는 함수로 정의하면 됨
# y의 truth를 true_f(train_data) 와 같이 구할 수 있도록

def true_f(x): return np.cos(x)  # <COGLINE>


# 위에서 작성한 Model 클래스 이용하여 model 객체 생성하기
# 하이퍼파라미터인 뉴런 개수 num_neurons=10 으로 시작

model = Model(num_neurons=10)  # <COGLINE>
```

학습시키기 전, 초기화된 상태의 model은 우리가 근사하고자 하는 목표 함수인 $f(x) = cos(x)$와 크게 다르다. 아래 코드를 완성하여 함수 $f$와 학습시키기 전의 근사함수 모델 $F$가 어떻게 다른지 살펴보자.

```python
# "True function: f" 와 근사함수인 "Approximating function: F"를 같은 축 상에 그래프로 나타내기
# 위에서 정의한 true_f 함수와 초기화된 model 객체를 잘 활용하기
# 두 그래프가 각각 어떤 그래프인지 알아보기 위해 label 설정은 필수, 색상 설정은 자유롭게

fig, ax = plt.subplots()
ax.plot(train_data, true_f(train_data), label="True function: f") # <COGLINE>
ax.plot(train_data, model(train_data).data, label="Approximating function: F") # <COGLINE>
ax.grid()
ax.legend();
```

이제 본격적으로 학습을 진행할 것이다. 하이퍼파라미터인 batch size, 학습률, epoch 수 등을 지정해준 상태에서 학습을 시작한다. 매 epoch마다 랜덤 batch 별 학습을 진행하기 위해서 batch를 먼저 정해주고, 각 batch에 대해 한 번에 학습을 진행하는 방식으로 프로그래밍하면 된다.

```python
# 하이퍼파라미터인 batch_size = 25 로 정하기
# 따라서 총 batch의 개수는 1000/25 = 40개

batch_size = 25 # <COGLINE>


# 하이퍼파라미터인 learning_rate = 0.01로 정하기

learning_rate = 0.01  # <COGLINE>


# 하이퍼파라미터인 epoch 수 1000으로 정하기
for epoch_cnt in range(1000):
    # 랜덤 batch 구성하기
    # train_data 배열을 실제로 섞을 필요는 없음
    # 대신 랜덤으로 섞인 index 배열을 가지고 있으면,
    # 그 index 배열에 따라 train_data 배열의 원하는 요소에 접근이 가능
    
    # 0부터 시작하는 index 배열 (idxs) 선언하기
    # 배열의 길이는 train_data의 길이와 같아야 함
    idxs = np.arange(len(train_data))

    # np.random.shuffle()을 이용하여 idxs 배열 랜덤하게 섞기
    # 반환하는 값 없이 입력된 배열을 바로 섞어 줌
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(train_data) // batch_size):
        
        # 랜덤하게 섞인 idxs 배열을 이용하여 batch_cnt번째 batch 구성하기
        # batch_cnt * batch_size 번째 idxs부터 (batch_cnt + 1) * batch_size - 1 번째 idxs까지를 사용하여 batch 구성하면 됨
        batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size] # <COGLINE>
        batch = train_data[batch_indices]

        # truth : batch에 대한 f(x) 값
        
        truth = true_f(batch)  # <COGLINE>
        
        
        # prediction : batch에 대한 F(x) 값
        
        prediction = model(batch)  # <COGLINE>
        

        # loss : 앞서 작성한 loss_func() 이용하여 truth와 prediction에 대한 loss 계산
        
        loss = loss_func(prediction, truth)  # <COGLINE>
        

        # 경사하강법 진행에 필요한 모든 편미분계수를 얻기 위해
        # loss에 대해 .backward() 메서드 실행
        
        loss.backward() # <COGLINE>

        
        # 앞서 작성한 gradient_step() 이용하여 model.parameters에 대해 경사하강 한 번 실행
        
        gradient_step(model.parameters, learning_rate)  # <COGLINE>
```

학습시킨 후의 model은 $f(x) = cos(x)$와 꽤 비슷해진다. 아래 코드를 완성하여 학습시킨 후의 근사함수 모델 $F$가 함수 $f$와 얼마나 가까워졌는지 살펴보자.

```python
# "True function: f" 와 근사함수인 "Approximating function: F"를 같은 축 상에 그래프로 나타내기
# 위에서 정의한 true_f 함수와 초기화된 model 객체를 잘 활용하기
# 두 그래프가 각각 어떤 그래프인지 알아보기 위해 label 설정은 필수, 색상 설정은 자유롭게

# <COGINST>
fig, ax = plt.subplots()
x = train_data
ax.plot(train_data, true_f(train_data), label="True function: f")
ax.plot(train_data, model(train_data).data, label="Approximating function: F")
ax.grid()
ax.legend();
# </COGINST>
```

### Step5. 학습 과정 다양하게 관찰하기


#### 학습 진행에 따른 손실의 변화를 실시간으로 관찰하기


학습이 진행됨에 따라 손실이 어떻게 변화하는지 확인하기 위해 noggin 라이브러리의 create_plot 함수를 사용해볼 것이다. 다음의 코드와 같이 create_plot() 함수는 plotter, fig, ax를 반환한다. 여기서 metrics=["loss"] 로 설정해주었기 때문에, plotter는 학습 과정을 진행하는 동안 2초마다 손실의 변화를 추적하는 역할을 한다.

```python
from noggin import create_plot

plotter, fig, ax = create_plot(metrics=["loss"])

# 그래프의 y축 값(loss)의 범위를 0과 1 사이 범위로 제한
# 초기 모델 F(x)가 어떻게 초기화되었는가에 따라 초기 loss가 지나치게 큰 경우도 발생할 수 있기 때문
ax.set_ylim(0, 1)
```

<!-- #region -->
이제 위에서 진행했던 학습을 다시 진행할 것이다. 그런데 이번에는 Step 4의 학습 코드의 적절한 위치에 아래의 두 코드를 넣어 다시 실행해보자. epoch를 반복하는 과정, batch 별 학습을 반복하는 과정이 진행되는 각 for문에 넣으면 된다. 그러면 학습 중에 loss를 추적할 수 있게 된다. 

```python
    plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)
```

```python
    plotter.set_train_epoch()
```
<!-- #endregion -->

<!-- #region -->
마지막으로 학습이 끝난 위치에 다음 코드를 추가해줌으로써, 누락된 데이터가 빠짐없이 기록되도록 할 수 있다.

```python
    plotter.plot()
```
<!-- #endregion -->

#### 근사함수 $F(x)$의 변화를 실시간으로 관찰하기


학습이 진행됨에 따라 모델 $F(x)$가 어떻게 변화하는지 확인하기 위해 matplotlib 라이브러리의 animation 모듈 내에 있는 FuncAnimation 함수를 사용해볼 것이다. $F(x)$가 $f(x)$에 점점 가까워지는 것을 확인할 수 있을 것이다. FuncAnimation 함수의 사용법은 딥러닝에 대해 배우는 데 필수적인 내용은 아니기 때문에, 주어지는 코드를 그대로 사용하여 그래프의 모습만 확인해보아도 충분하다

<!-- #region -->

그런데, 아래에 주어질 코드를 사용하기 위해서는 학습 과정 중의 모델 파라미터 쌍들을 여러 차례 저장한 params 리스트를 준비해야 한다. 따라서 Step 4의 학습 코드에서 적절한 위치를 찾아 아래 두 코드를 넣은 후 다시 학습을 진행해야 한다.

```python
# 모델 파라미터 쌍들을 저장하는 리스트 선언
params = [] 
```

```python

# 매 epoch 마다 저장하기에는 너무 많으므로, 10 epoch 마다 저장
# model.parameters 속성을 이용하여 모델 파라미터를 불러옴
# 10의 배수인 epoch이면 params 리스트에 그 순간의 모델 파라미터 append 
if epoch_cnt % 10 == 0:
    params.append([p.data.copy() for p in model.parameters])
```
<!-- #endregion -->

이제 params 리스트까지 준비가 되었다면, 아래의 코드를 그대로 실행하여 근사함수 $F(x)$가 함수 $f(x)$에 어떻게 가까워져 가는지 확인할 수 있다.

```python
from matplotlib.animation import FuncAnimation

x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)

fig, ax = plt.subplots()
ax.plot(x, np.cos(x))
ax.set_ylim(-2, 2)
_model = Model(model.N)
_model.load_parameters(*params[0])
(im,) = ax.plot(x.squeeze(), _model(x[:, np.newaxis]).squeeze())

def update(frame):
    # ax.figure.canvas.draw()
    _model.load_parameters(*params[frame])
    im.set_data(x.squeeze(), _model(x[:, np.newaxis]).squeeze())
    return (im,)

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(params)),
    interval=20,
    blit=True,
    repeat=True,
    repeat_delay=1000,
)
```

#### 각 뉴런 별 학습 결과 그래프로 나타내기


이번에는 학습을 진행한 $N=10$개의 뉴런으로부터 얻는 각각의 결과값을 그래프로 나타내어 보자. 즉, $i=1, ..., N$의 N개의 뉴런에 대해 구간 $x \in [-2\pi, 2\pi]$에서 $\varphi(x \cdot w_{i} + b_{i})$의 그래프가 어떻게 나타나는지 그래프로 그려보면 된다. 각 뉴런의 결과까지만 확인하는 것이므로 이 식에서 $v_i$는 미포함한다.


$N=10$개의 뉴런으로부터 얻는 결과값을 하나의 그래프로 나타내기 위해, subplot이 열 ncols=2, 행 nrows=model.N//2 개인 그래프를 생성할 것이다. 따라서, axes는 원소가 2*(N//2)개인 배열이 된다. 또한 model.parameters[0] (배열 $\{w_i\}$)와 model.parameters[1] (배열 $\{b_i\}$)은 원소가 $N$개인 배열이다. $N$개의 뉴런에 대응되는 각각의 축에 대해, 정의역 x와 각각의 모델 파라미터들로부터 얻어지는 (x, sigmoid(x * w_i + b_i))를 plot하면 우리가 원하는 결과를 얻을 수 있다.


다음의 코드를 완성하여 원하는 N=10개의 그래프를 그려보자.

참고로, 코드에 나온 flatten() 함수는 다차원 배열을 1차원으로 풀어쓰는 역할을 해주는 함수로, 이론에서 잠깐 나왔던 벡터화(Vectorization)를 수행하는 함수이다. 또한 zip() 함수는 Python의 내장함수로, 여러 개의 iterable한 객체들이 전달되었을 때 각 객체의 요소들을 순서대로 하나씩 뽑아 튜플의 형태로 묶어주는 역할을 한다. 특히 for문에서 활용하기 유용한 함수이다. flatten()과 zip() 함수를 꼭 이용해야만 구현할 수 있는 것은 아니지만, 알고 있으면 더 쉽게 프로그래밍하는 데 도움이 될 것이다.

```python
fig, axes = plt.subplots(ncols=2, nrows=model.N // 2)
x = np.linspace(-2 * np.pi, 2 * np.pi) # 디폴트 : 50개 구간

for ax, w, b in zip(axes.flatten(), model.parameters[0].flatten(), model.parameters[1].flatten()):
    ax.plot(x, sigmoid(x * w + b)) # <COGLINE>
    ax.grid("True")
    ax.set_ylim(0, 1)
```

마지막으로 각 뉴런에서 얻어진 결과값에 $v_i$를 곱해준 값들을 하나의 그래프에 plot 하고, 이들을 모두 합한 것을 굵은 검은 점선으로 plot 해보자.  즉, $i=1, ..., N$의 N개의 뉴런에 대해 구간 $x \in [-2\pi, 2\pi]$에서 $v_i\varphi(x \cdot w_{i} + b_{i})$의 그래프가 어떻게 나타나는지 그래프로 그려보면 된다. 그리고 이들을 모두 합한 그래프를 함께 그려서, 이 그래프의 모양이 함수 $f(x)=cos(x)$와 얼마나 가까운지 확인해보자.

```python
fig, ax = plt.subplots()
x = np.linspace(-2 * np.pi, 2 * np.pi)
F = mg.linspace(0,0)

# 하나의 축 위에 모든 그래프를 plot
W = model.parameters[0]
B = model.parameters[1]
V = model.parameters[2]
for w, b, v in zip(W.flatten(), B.flatten(), V.flatten()):
    ax.plot(x, v * sigmoid(x * w + b)) # <COGLINE>
    F += v * sigmoid(x * w + b)
    
# v * sigmoid(x * w + b)를 모두 합한 것을 함께 plot
ax.plot(
    x,
    F,
    color="black",
    ls="--",
    lw=4,
    label="full model output",
)

ax.legend()
ax.set_xlabel(r"$x$")
```

### 배운 내용 되돌아보기


이번 실습에서는 Universal Approximation Theorem에 의거하여 함수 의 근사함수 의 모델을 로 세우고 학습을 진행하였다. 우리가 학습시킨 모델은 입력층과 출력층의 뉴런 수가 모두 1인 매우 간단한 형태의 신경망 모델이었지만, 앞으로 작성할 다양한 형태의 신경망의 기본이 되는 중요한 개념들이 많이 나왔다.

- batch의 개념을 적용하여 밀집층의 순전파를 직접 프로그래밍하였다. 근사함수 수식을 코딩으로 구현한 결과는 딱 한 줄이었지만, 그 한 줄이 잘 성립하는 코드인지를 확인하기 위해 배열의 연산을 하나씩 따라가며 자세히 살펴보았다.

- 이렇게 작성한 순전파 코드를 바탕으로 모델(클래스)를 정의하였다. 하이퍼파라미터인 은닉층의 뉴런 개수는 사용자가 입력하도록 하였고, 모델 파라미터는 표준정규분포를 따르도록 초기화하였다.

- 손실함수를 잔차 절댓값의 평균으로 정의하였다. batch에 대해 평균적인 손실을 구하여 학습에 사용함으로써 batch의 이점을 한 번 더 확인하였다.

- 경사하강법 함수는 실습 3과 실습 4에서 작성했던 함수와 동일하게 다시 작성하며 복습하였다.

- 근사하고자 하는 목표 함수가 로 명확히 주어졌기 때문에,  데이터 (train_data) 만 정해주면 함수를 이용하여 데이터셋의  데이터 (truth)를 쉽게 얻을 수 있었다. 이때,  데이터들에 대한 편미분 계수가 계산되지 않도록 를 상수 취급해야 했다.

- 학습을 진행할 때는 랜덤 batch를 구성하여 각 batch에 대해 iteration을 진행해야 했다. 이때, 랜덤 batch는 랜덤하게 섞인 인덱스 배열을 이용하여 train_data 배열에 랜덤하게 접근하는 방식으로 사용하였다.

- 1 iteration 동안 실제(truth)값과 모델의 순전파로 얻은 예측(prediction)값을 손실함수에 대입한 후, 손실함수에 대해 자동미분을 이용한 경사하강법을 진행하여 모델 파라미터를 1회 업데이트하였다. 모든 batch에 대해 iteration을 진행하는 epoch를 총 1000회 진행함으로써 모델을 효과적으로 학습시켰다.

- 학습 전과 후의 목표 함수 와 근사함수 를 비교해보았고, 학습이 진행됨에 따라 가 에 가까워지는 과정을 관찰해보기도 했다.

- 학습이 진행됨에 따라 손실 loss가 실시간으로 감소하는 것을 noggin 라이브러리를 이용하여 관찰하였다.

- 학습이 완료된 후, 은닉층의 각 뉴런들이 어떤 결과를 도출하는지 확인해보았다. 또한, 각 뉴런에서 도출한 결과에 가중치 연산을 한 결과와 그렇게 구한 모든 함수를 합한 결과를 함께 시각화함으로써 뉴런 단위로 학습 결과가 어떻게 나타났는지 살펴보았다.
