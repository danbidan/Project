---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="qJ2eOwDi3Y58" -->
# **실습11. Attention 기반 seq2seq**  

> 마크다운 파일은 실습 위주로 다룰 것이므로, attention에 대한 개념 이해는 본책 3권의 '0. basic seq2seq의 한계', '1. 도식으로 Attention의 원리 이해하기' 파트를 활용하도록 하자. 본 마크다운에서는 '2. 수식과 핵심 코드로 attention 개념 심화하기'로 넘어가볼 것이다.
<!-- #endregion -->

<!-- #region id="by5fko1j31W9" -->
## **2. 수식과 핵심 코드로 attention 개념 심화하기**

### **2.1 수식으로 attention score 개념 다지기**

attention 메커니즘의 핵심은 바로 attention score이다. attention score란, 디코딩 과정에서 인코더의 어느 입력 step에 주목할 것인지의 가중치를 점수로 나타낸 값을 말한다. 디코딩 과정이 t개의 step들로 나뉘어 있다고 생각해보자. (한 step마다 하나의 output을 만든다.) 앞서 언급했듯, attention scores는 ‘인코더의 모든 hidden states’와 ‘현재 디코더의 hidden descriptor’ 두 가지를 모두 고려하여 계산된다. 현재 decoder가 t번째 step에 있다면, attention scores(=$e_t$)의 계산에 필요한 재료들은 이하와 같다. 


---



**attention score** =  $\begin{equation}
\vec{e}_t = H^e W_\alpha \vec{h}{}^d_t
\end{equation}$  
>(1) $H^e$: 인코더의 모든 hidden descriptor 모음
>D-차원의 모든 T개의 인코더 hidden descriptors 행렬 (T,D)  
$\begin{align}
&\:\begin{matrix}\xleftarrow{\hspace{0.75em}} & D & \xrightarrow{\hspace{0.75em}}\end{matrix} \\
H^e =\;\, &\begin{bmatrix}\leftarrow & \vec{h}{}^e_1 & \rightarrow \\ \leftarrow & \vec{h}{}^e_2 & \rightarrow \\ \vdots & \vdots & \vdots \\ \leftarrow & \vec{h}{}^e_T & \rightarrow\end{bmatrix}\;\;\begin{matrix}\bigg\uparrow \\ T \\ \bigg\downarrow\end{matrix}
\end{align}$  

>(2) $W_a$: 학습 가능한 매개변수인 가중치에 대한 행렬 (D,D)  
>$\begin{align}
&\;\begin{matrix}\xleftarrow{\hspace{2.25em}} & D & \xrightarrow{\hspace{2.25em}}\end{matrix} \\
W_\alpha =\;\, &\begin{bmatrix}\uparrow & \uparrow & \cdots & \uparrow \\ \vec{W}_1 & \vec{W}_2 & \cdots & \vec{W}_D \\ \downarrow & \downarrow & \cdots & \downarrow\end{bmatrix}\;\;\begin{matrix}\big\uparrow \\ D \\ \big\downarrow\end{matrix}
\end{align}$  

>(3) ${h_t}^d$: $t$ step에서의 디코더 hidden descriptor (D-차원)
<!-- #endregion -->

<!-- #region id="2mCwdKMW5kLT" -->
위 수식을 달리 설명하면, t step에서의 attention scores인 $e_t$는 ‘${h_t}^d$가 확장된 형태’라고도 볼 수 있다. 이때 디코더의 hidden descriptor는 인코더의 모든 descriptors를 검토한 후 학습을 통해 자신과 가장 ‘연관 있는’ 인코더 토큰이 무엇인지 판단한다. 그리고 모델은 학습을  더 많이 진행함에 따라 ${h_t}^d$와 각 인코더 hidden descriptor 간의 관련도를 더 정확하게 파악하게 된다.
<!-- #endregion -->

<!-- #region id="RMjVNSd-5sXt" -->
### **2.2 디코더 hidden descriptor의 학습 과정 이해하기**

attention score 계산 수식인 $e_t = H^eW_a{h_t}^d$을 기반으로, ${h_t}^d$가 학습을 통해 어떻게 개선되는지 그 과정을 수식과 코드로 구체적으로 알아보자.  

 ① $H^eW_\alpha$에 대한 행렬곱부터 시작해보자. $H^eW_\alpha$의 각 value는 인코더 hidden state와 가중치 벡터 간의 ‘유사성'을 나타내게 된다.  

 $\begin{align}
&\;\begin{matrix}\xleftarrow{\hspace{4.75em}} & D & \xrightarrow{\hspace{4.75em}}\end{matrix} \\
H^eW_\alpha =\;\, &\begin{bmatrix}\vec{h}{}^e_1\cdot\vec{W}_1 & \vec{h}{}^e_1\cdot\vec{W}_2 & \cdots & \vec{h}{}^e_1\cdot\vec{W}_D \\ \vec{h}{}^e_2\cdot\vec{W}_1 & \vec{h}{}^e_2\cdot\vec{W}_2 & \cdots & \vec{h}^e_2\cdot\vec{W}_D \\ \vdots & \vdots & \ddots & \vdots \\ \vec{h}{}^e_T\cdot\vec{W}_1 & \vec{h}{}^e_T\cdot\vec{W}_2 & \cdots & \vec{h}{}^e_T\cdot\vec{W}_D\end{bmatrix}\;\;\begin{matrix}\bigg\uparrow \\ T \\ \bigg\downarrow\end{matrix}
\end{align}$

<!-- #endregion -->

```python id="pgVFKnPQ50PH"
## H^eW_a = W_\alpha @ H^e
## 코드로 적용하면?

precomputed_encoder_score_vectors = W_alpha(encoder_hidden_states)
# (T, N, D) @ (D, D) -> (T, N, D)

""" 
위의 방법을 통해, 디코더를 호출할 때마다 다시 계산을 반복하는 것이 아니라, 단 한 번만으로 값들을 계산할 수 있게 된다.
각 디코더 단계 동안, 내적을 통해 최종 attention score 벡터를 계산할 수 있다.
""" 

```

<!-- #region id="4ZYXyYqp52Mq" -->
② ${h_t}^d$에 유사성 점수 $H^eW_\alpha$를 행렬곱해주면, 디코더 hidden state의 각 구성 요소에 가중치를 반영할 수 있게 된다. 이렇게 세 가지 요소가 모두 곱해진 $H^eW_a{h_t}^d$는 attention score로 기능하게 된다.
<!-- #endregion -->

<!-- #region id="3TpRvQAt6ElZ" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="ZxkfyOMk6F16" -->
$e_t$ 행렬에서 $i$번째 행은 attention score $e_{t,i}$가 되며, 이때 $e_{t,i}$는 $i$번째 인코더 hidden state와  ${h_t}^d$(=현재 디코더 hidden descriptor)가 서로 관련된 정도 즉 유사도를 표현해주게 된다. 계산 과정을 풀어보자면, 예컨대 $e_{t,1}$을 계산하려면 ${h_1}^e$와 W를 행렬곱한 값에다 D차원의 ${h_t}^d$를 하나씩 곱해줌으로써 구할 수 있다.
<!-- #endregion -->

```python id="3qJr_-286R7T"
## e_t = {H^eW_a * {h_t}^d}.sum(axis=-1) 
## 코드로 적용하면?

e_t = (precomputed_encoder_score_vectors * h_t).sum(axis=-1)
# (T, N, D) * (1, N, D) -> (T, N, D).sum(axis=-1) -> (T, N)
```

<!-- #region id="5zQuuAKm6YrV" -->
>잠깐! 3차원 데이터를 다룰 때 데이터의 방향에는 x축 방향, y축 방향, z축 방향이 있다. numpy에서 axis=0는 x축 방향, axis=1는 y축 방향, axis=2은 z축 방향을 가리키며, 3차원 데이터에서 axis=-1는 곧 axis=2 즉 z축 방향을 의미한다. 따라서, 위 수식에서 ‘sum(axis=-1)’란, 3차원 데이터에서 z축 방향으로 같은 위치에 놓인 value끼리 합해준다는 것을 의미한다.
<!-- #endregion -->

<!-- #region id="ySMUL3yC6cvP" -->
이 attention scores 벡터를 사용하면, 디코더 hidden descriptor에 대한 attention weights를 계산할 수 있다. 만약 어떠한 인코더 state의 weight가 특히 높다면, "현재 디코더 step의 경우, 다른 hidden states보다 해당 인코더 state에 특히 주목해야 한다"는 정보로 이해할 수 있다. 같은 방식으로, 낮은 가중치값에 대해서는 "현재 디코더 단계의 경우, 이 특정한 인코더 hidden state에 포함된 정보를 대부분 무시해도 된다."라는 의미로 이해하면 된다.

이러한 ① → ②의 방식으로 학습 과정을 반복하면서, 모델은 각 디코더 hidden descriptor에게 각 인코더 hidden state의 중요성을 부여하는 방법을 학습하게 된다. 다시 말해, 모든 인코더 hidden states 가운데 특정 i번째 디코더 hidden state와 가장 ‘관련 있는’ 인코더 hidden state가 무엇인지 더 잘 판단할 수 있도록, i번째 가중치 벡터가 조율된다고 요약할 수 있다. 만약 특정 인코더 hidden state가 어떤 디코더 hidden state(D차원)에 대해 D개의 항 전반에 걸쳐 높은 관련도를 가지고 있다면, 가중치 벡터는 이러한 정보를 학습한 후 attention score을 통해 증폭시킬 것이다. 이렇게 학습을 통해 개선된 가중치 행렬을 통해 모델은 인코더의 hidden descriptors와 디코더의 hidden descriptors 사이의 더 복잡한 문맥적 관계를 파악할 수 있게 된다.


<!-- #endregion -->

<!-- #region id="Jl-Nh0e96fKI" -->
### **2.3 attention weight 계산하기**

실제로 문맥 벡터를 구성하는 값은 attention score가 아니라 attention score를 확률값 형태로 표현한 attention weight이다. 이때 필요한 것은 소프트맥스 함수이다. 우리는 소프트 맥스 함수에 (현재 디코더 hidden descriptor에 대한) attention score을 입력하여, 전체에 대한 확률값(0과 1사이의 값) 형태인 attention weight를 구할 수 있다.


$\alpha_t=softmax(e_t)=\frac{exp(e_t)}{\sum_{i=1}^{T}exp(e_{t,i})}=\frac{exp(e_t)}{exp(e_{t,1})+exp(e_{t,2})+\cdots+exp(e_{t,T})}$  


편리하게도, MyGrad에는 소프트맥스 기능이 내장되어 있다! 단, 소프트맥스를 attention score 행렬에 취할 때, score에 해당되는 축에서 이루어져야 함을 명심하자.  
<!-- #endregion -->

```python id="GmJYnDic6rAi"
## attention weight = softmax(attention score) 
## 코드로 적용하면?

a_t = mg.nnet.softmax(e_t, axis=0)
# softmax((T, N), axis=0) -> (T, N)

```

<!-- #region id="jeP5E6AE6tft" -->
### **2.4 문맥벡터 계산하기**
마지막으로, 우리는 모든 인코더 hidden states의 정보를 '요약'해주는 문맥 벡터 $c_t$를 계산하기 위해 방금 계산한 attention weights를 사용할 것이다. 아래 식에서 알 수 있듯 attention weights $\alpha_j$는 각 인코더 hidden state ${h_j}^e$의 계수가 되며, $\alpha_j$와 ${h_j}^e$에 대한 가중합이 문맥벡터로 기능하게 된다.  

$c_t=\sum_{j=1}^{T}\alpha_j{h_j}^e=\alpha_1({h_1}^e)+\alpha_2({h_2}^e)+\cdots+\alpha_T({h_T}^e)$  

관련성이 상대적으로 적은 인코더 hidden states에는 낮은 attention weights가 주어지게 된다. 따라서, 문맥 벡터는 주로 가장 관련성이 높은 hidden states의 정보를 포함하게 될 것이다. 이를 구현할 때, 1번째 축(N)이 배치 차원 즉 배치 내 시퀀스의 개수를 나타내므로 이에 유의하여 시퀀스 길이를 나타내는 0번째 축(T)을 따라 총합을 계산해야 함을 기억하라.

<!-- #endregion -->

```python id="oGtP3ZnR6yig"
## context vector = {\alpha_j * {h_j}^e}의 가중합
## 코드로 적용하면?

c_t = (a_t[..., None] * encoder_hidden_states).sum(axis=0, keepdims=True)
# (T, N, 1) * (T, N, D) -> (T, N, D).sum(axis=0, keepdims=True) -> (1, N, D)

```

<!-- #region id="k2ohgJx263OG" -->
### **2.5 문맥벡터를 디코더에 넘겨주기**

지금까지 문맥벡터 과정을 훌륭하게 이해해보았다. 이제는 문맥 벡터를 어떻게 다루어야 할지 고민해야 한다. 어떤 방식을 통해 디코더에 문맥 벡터와 디코더의 output을 둘 다 입력해줄 수 있을까? 기계학습 연구원들이 해온 방식에 여러 가지가 있는데, 그 중 우리가 취해볼 접근법은, 문맥 벡터 $c_t$를 일반적인 디코더 단계의 output $y_t$과 연결하는(concatenate) 방식이다. 즉, 우리는 디코더 단계에 대한 계산을 $s_t$와 ${h_t}^d$를 사용하여 평소대로 진행하고 $y_t$와 ${h_t}^d$를 얻어낼 것이다. 우리는 $y_t$ 값을 저장한 후, 다음 디코더의 input $s_{t+1}$에는 $y_t$와 문맥 벡터 $c_t$를 이어 붙인 값을 넣어줄 것이다 

<!-- #endregion -->

```python id="t6oYaqkF661i"
## 문맥벡터를 디코더에 넘겨주기 = 문맥벡터와 디코더 output을 연결
## 코드로 적용하면?

y_and_c = mg.concatenate([y_t, c_t], axis=-1)
# concatenate([(1, N, K), (1, N, D)], axis=-1) -> (1, N, K + D)

```

<!-- #region id="56lLu1Ho676X" -->
우리는 output 벡터의 차원을 변경하기 때문에, dense layer를 사용하여 output 차원을 수정하고 최종 분류 점수를 재계산해야 한다.
<!-- #endregion -->

```python id="ncFKCtv-69cQ"
y_t = post_concat_dense(y_and_c)
# (1, N, K + D) @ (K + D, K) -> (1, N, K)
```

<!-- #region id="BBgRj2fO6-gu" -->
이게 전부이다! 마지막 dense layer는 일종의 인터프리터 역할을 하는데, 문맥 벡터에서 전달되는 정보와 디코더 output의 정보를 통합해준다. 따라서 각 디코더 단계에 대한 최종 ouput에는 가장 중요한 인코더 hidden states로부터 직접 가져온 정보가 포함된다. 
<!-- #endregion -->

<!-- #region id="ST9vRU9C7ANa" -->
## **3. 코드로 seq2seq with attention 구현하기**

이제는, 이 모든 것들을 실행에 옮겨보자. 다행히 이전에 만들어두었던 RNN 클래스를 활용하면, 우리가 현재 필요로하는 텐서, 즉 모든 hidden states를 포함하는 텐서를 반환할 수 있다. 덕분에 우리는 $W_\alpha$에 대한 MyNN  <dense>만 잘 정의해주면, attention scores를 계산할 수 있게 되었다. 이때 $H^eW_a$에 대한 계산은, 인코더의 output을 받은 직후, 어떤 다른 decoder 단계가 진행되기 전에 이루어져야 함을 명심하자.

아래 클래스에서 위에서 설명한 attention mechanism을 적용하여 이전의 seq2seq 모델을 강화해보자. 각 디코더 단계에서는 계산된 attention weights를 저장하라. __call__에서는 예측 점수 및 attention weights에 대한 행렬(T,T,N) 을 모두(두 개) 반환하라. 

- 이때 (T,T,N)은 데이터를 이루는 시퀀스의 개수이며, 앞의 T,T는 데이터에 대한 길이를 의미한다. 차원이 2번 필요한 것은 각 토큰(T)별로 어떤 토큰(T)이 연관이 많은지 따져야 하기 때문이다. 구체적으로 보면 에서 첫 번째 T는 각 디코더 토큰이 인코더의 어떤 토큰과 관련 있는지(=attention weights)를 나타내주고, 두 번째 T는 디코더 토큰의 개수를 나타낸다. 둘의 의미를 혼동하지 않도록 하자. 

<!-- #endregion -->

```python id="3fwehwqx7G1N"
class Attentionseq2seq:
    def __init__(self, dim_input, dim_recurrent, dim_output):
        """ Initializes all RNN layers needed for seq2seq
        
        매개변수
        ----------
        dim_input: int 
            RNN을 지나가는 데이터의 차원 (C)
        
        dim_recurrent: int
            RNN에 있는 hidden state의 차원 (D)
        
        dim_output: int
            RNN의 output의 차원 (K)
        

        Notes
        -----
        이 특정 문제에 대해, input의 차원과 output의 차원은 동일(C = K).
        그러나 이것이 일반적인 경우는 아님.
        """
        # (이전에 했던 것처럼) 두 개의 RNNs - 인코더와 디코더 -를 생성(인스턴스화)
        #
        # 이후, 두 개의  MyNN dense layer 생성:
        # - 하나는 (D, D) 행렬 W_alpha에 해당되는 것 (bias는 없음),
        # - 하나는 y_t 벡터와 c_t 벡터가 concat된 것 (dim K+D)
        #   적절한 output dimension (dim K)을 갖춘 벡터.
        #
        # 두 개의 dense layers 모두, glorot_normal weight initializer 사용
        # <COGINST>
        self.encoder = RNN(dim_input, dim_recurrent, dim_output)
        self.decoder = RNN(dim_input, dim_recurrent, dim_output)
        
        self.W_alpha = dense(dim_recurrent, dim_recurrent, weight_initializer=glorot_normal, bias=False)
        self.post_concat_dense = dense(dim_recurrent + dim_output, dim_output, weight_initializer=glorot_normal)
        # </COGINST>


   def __call__(self, x):
        """ seq2seq에 대한 완전한 순방향 패스(full forward pass)를 수행.
        
        매개변수
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(T, N, C)
            batch에 있는 각 시퀀스에 대한 원-핫 인코딩
        
        반환 값
        -------
        y: mygrad.Tensor, shape=(T, N, K)
            각 디코더 step의 output으로부터 산출된 최종 분류 점수
        a_ij: numpy.array, shape=(T, T, N)
            sequences의 batch에 대한 attention weights. The 0-th
            0차원은 각 디코더 step에서 계산된 attention weights에 상응함
        """
        # 인코딩 및 디코더 setup을 `seq2seq` 모델에서와 동일하게 수행:
        #
        # - 인코더 hidden states `enc_h`를 받음,
        # - 첫 번째 디코더 hidden state가 마지막 인코더 hidden state가 되도록 설정
        # - 리스트 `y`를 생성하여 디코더 outputs를 저장
        # - <START> 토큰 원-핫 인코딩을 첫 번째 디코더 input으로 초기설정
        # <COGINST>
        T, N, C = x.shape
        _, enc_h = self.encoder(x) # enc_h: shape-(T, N, D)
        
        h_t = enc_h[-1:] # h_t: shape-(1, N, D)
        y = []
        
        s_t = np.zeros((1, N, C), dtype=np.float32)
        s_t[..., -2] = 1
        # </COGINST>
        
        # W_alpha에 상응하는 dense layer를 `enc_h`에 적용하여
        # attention scores의 일부를 미리 계산
        # <COGINST>
        attn_score_precomputed = self.W_alpha(enc_h) # context: shape-(T, N, D)
        # </COGINST>
        
        # 아래 코드는 `a_ij` 리스트를 생성
        # 리스트에 각 디코더 step에서의 attention weights를 저장
        a_ij = []


for _ in range(T):
            # `e_t`:현재 디코더 hidden state에 대한 attention scores 계산
            # 사용 수식: 미리 계산된 W_alpha * H^e. 
            # <COGINST>
            e_t = (h_t * attn_score_precomputed).sum(axis=-1) # e_t: shape-(T, N)
            # </COGINST>
            
            # attention weights: attention scores에 소프트맥스 함수를 취한 것
            # `T` 차원에 대해 softmax 함수를 취할 것을 명심하라.
            #
            # 결과를 변수 `a_t`에 저장.
            # <COGINST>
            a_t = mg.nnet.softmax(e_t, axis=0) # a_t: shape-(T, N)
            # </COGINST>
            
            # 아래 코드는 현재 디코더 step에 대한 attention weights를 `a_ij` 리스트에 더해줌
            # 새로운 축은 weights가 하나의 행렬로 연결(concatenate)될 수 있도록 만듦
            a_ij.append(a_t.data[None])

           # 인코더 hidden states로부터 문맥 벡터 `c_t`를 계산하라 
            # `enc_h` & the attention weights `a_t`.
            # <COGINST>
            c_t = (enc_h * a_t[..., None]).sum(axis=0, keepdims=True) # c_t: shape-(1, N, D)
            # </COGINST>
            
            # basic seq2seq 모델에서처럼, 
            # 하나의 디코더 step를 수행하여 y_t와 h_t를 입력받아라
            # <COGINST>
            # y_t: shape-(1, N, K)
            # h_t: shape-(1, N, D)
            y_t, h_t = self.decoder(s_t, h_t)
            # </COGINST>

            # 마지막 축을 따라 디코더 output y_t와 문맥벡터 c_t를 연결(concatenate)하라            
            # (1, N, K)와 (1, N, D)를 연결한 결과 (1, N, K+D)의 shape를 가짐
            # <COGINST>
            y_t = mg.concatenate([y_t, c_t], axis=-1) # y_t: shape-(1, N, K + D)
            # </COGINST>
            
            # dense layer를 활용하여
            # 연결한(concatenated) 벡터를 -> 적절한 output 차원(K)의 벡터로 압축시켜라
            # 리스트 `y`에 마지막 `y_t`를 추가하라
            # <COGINST>
            y_t = self.post_concat_dense(y_t) # y_t: shape-(1, N, K)
            y.append(y_t)
            # </COGINST>
            
            # `one_hot_encode_prediction`를 사용하여 
            # 디코더 output과 문맥벡터 `y_t`의 결합물을 기반으로
            # 다음 디코더 input s_{t+1}을 찾아내라
            # <COGINST>
            s_t = one_hot_encode_prediction(y_t) # s_t: shape-(1, N, K)
            # </COGINST>
        
        # 0-th axis을 따라
        # `y`에 저장된 y_t 텐서들을 모두 연결(concatenate)하라
        # <COGINST>
        y = mg.concatenate(y, axis=0) # y: shape-(T, N, K)
        # </COGINST>
        
        # 아래 코드를 통해
        # 각 디코더 step에 대해 계산된 T개의 attention vectors `a_t`를 연결할 수 있다
        a_ij = np.concatenate(a_ij, axis=0) # a_ij: shape-(T, T, N)
        
        # 위 docstring에 부합하는 적절한 텐서들과 배열들을 반환하라
        return y, a_ij # <COGLINE>

@property
    def parameters(self):
        """ 모델 내 모든 매개변수를 취하는 편리한 함수
        이것은 `model.parameters`를 통해 속성(attribute)으로 액세스 가능
        
        반환 값
        -------
        Tuple[Tensor, ...]
            우리 모델에 대한 모든 학습 가능한 매개변수를 포함하는 튜플
        """

        # <COGINST>
        return (self.encoder.parameters + self.decoder.parameters
                + self.post_concat_dense.parameters + self.W_alpha.parameters)
        # </COGINST>

```

<!-- #region id="umtrjHRc7S00" -->
여기에 noggin plot을 그려(인스턴스화) 모델의 손실과 정확성을 추적하라.  
<!-- #endregion -->

```python id="CMDw3tO57Tjl"
from noggin import create_plot
plotter, fig, ax = create_plot(["loss", "accuracy"]) # <COGLINE>
```

<!-- #region id="VxZoj9JW7V4S" -->
Attentionseq2seq 모델과 Adam 옵티마이저를 설치하라. 모델의 경우 dim_recurrent가 25인 것이 좋다. 옵티마이저의 경우 매개변수를 default로 지정하는 것이 시작하는 단계에서 적합할 것이다.
<!-- #endregion -->

```python id="ak7hd_UQ7W6M"
# <COGINST>
model = AttentionSeq2Seq(dim_input=12, dim_recurrent=25, dim_output=12)
optimizer = Adam(model.parameters)
# </COGINST>
```

<!-- #region id="u48vw6UB7YOk" -->
아래에 학습 loop를 작성하라. 필요에 따라 inputs를 softmax_crossentropy로 reshape해야 함을 기억하라. 이때 배치 크기는 100으로 지정하고, 시퀀스 길이는 1에서 20 사이가 되도록 하라. 모델을 8000회 반복하여 학습시켜라. 
<!-- #endregion -->

```python id="TFZKIGp37Yt0"
# <COGINST>
batch_size=100

for k in range(8000):
    x, target, sequence = generate_batch(batch_size=batch_size)

    output, _ = model(x)
    
    loss = softmax_crossentropy(output.reshape(-1, 12), target.reshape(-1))
    loss.backward()
    optimizer.step()
    
    acc = np.mean(np.argmax(output, axis=-1) == target)

    plotter.set_train_batch({"loss":loss.item(), "accuracy":acc}, batch_size=batch_size)
    
    if k % 500 == 0 and k > 0:
        plotter.set_train_epoch()

plotter.plot()
# </COGINST>
```

<!-- #region id="YCHgijKh7aVI" -->
다음을 실행하여 모형의 정확성을 평가하라.
<!-- #endregion -->

```python id="cpFPOsdl7b8R"
length_total = defaultdict(int)
length_correct = defaultdict(int)

with mg.no_autodiff:
    for i in range(50000):
        if i % 5000 == 0:
            print(f"i = {i}")
        x, target, sequence = generate_batch(1, 20, 1)

        output, _ = model(x)

        length_total[sequence.size] += 1
        if np.all(np.argmax(output, axis=-1) == target):
            length_correct[sequence.size] += 1

fig, ax = plt.subplots()
x, y = [], []
for i in range(1, 20):
    x.append(i)
    y.append(length_correct[i] / length_total[i])
ax.plot(x, y);
```

<!-- #region id="VKcSgZim7dh3" -->
이제 우리는 학습한 모든 시퀀스 길이에 대해 거의 완벽에 가까운 정확성을 확인할 수 있다. (충분히 오래 학습한 경우, 모델은 이 문제를 완전히 마스터할 수 있다).

마무리 단계로, 단일 sequence에 대해 계산된 attention weights를 살펴보자. 아래 셀을 실행하면, 선택한 시퀀스 길이에 대한 attention weights를 시각화할 수 있다.
<!-- #endregion -->

```python id="Y0x-t36k7gF4"
seq_len = 15

x, target, sequence = generate_batch(seq_len, seq_len, 1)
y, a_ij = model(x)
y = np.argmax(y, axis=-1).squeeze() # determine decoder inputs

fig, ax = plt.subplots()

ax.set_yticks(range(y.size))
ax.set_yticklabels(["<S>"] + [x for x in y[:-1]])
ax.set_ylabel("Input to Decoder")

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

ax.set_xticks(range(target.size))
ax.set_xticklabels([x for x in sequence.squeeze()] + ["<E>"])
ax.set_xlabel("Original Sequence")

ax.imshow(a_ij.squeeze());
```

<!-- #region id="sXAORxA67gtb" -->
이런 결과가 나올 것이라 예상하였는가? 우리가 해결하려는 문제를 고려할 때, 직관적으로 이해가 잘 되는가?

사실상 딥러닝 분야에서 attention의 도입됨으로써 딥러닝을 이해하는 데 중요한 진전을 이루어낼 수 있었다. 실제로 attention은 매우 강력한 도구이기 때문에, 마지막 노트에서는 어떻게 우리가 attention만을 활용하여 언어 모델을 구성할 수 있는지 알아보도록 하자.  
<!-- #endregion -->
