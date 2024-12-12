import random
from time import time
from math import pi,sqrt

err=1e-7 #오차범위

def taylor_atan(x,i):  # atan(x)의 테일러 급수의 i+1번째 항
    return (-1)**i*x**(2*i+1)/(2*i+1)

    
# 몬테카를로법
count=0
result=0
i=0
start=time()

while abs(pi-result)>err:
    x=random.random()
    y=random.random()
    if x*x+y*y<1:
        count+=1
    i+=1
    result=4*count/i
print(f'0. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')




#라이프니츠

result=0
i=0
start=time()
while abs(pi-result*4)>err:
    result+=(-1)**i/(2*i+1)
    i+=1

result*=4
print(f'1. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')


# 오일러
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=4*(taylor_atan(1/2,i)+taylor_atan(1/3,i))
    i+=1
print(f'2. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')



#해튼
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=12*(taylor_atan(1/4,i))+4*taylor_atan(5/99,i)
    i+=1
print(f'3. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')



#마틴
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=16*taylor_atan(1/5,i)-4*taylor_atan(1/239,i)
    i+=1
print(f'4. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')


#뉴턴
factor=1
result=0
i=1
start=time()
while abs(pi-result*6)>err:
    result+=factor*0.5**(2*i-1)/(2*i-1)
    factor*=(2*i-1)/(2*i)
    i+=1
result*=6
print(f'5. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i}')


#베가
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=16*taylor_atan(1/5,i)-8*taylor_atan(1/408,i)+4*taylor_atan(1/1393,i)
    i+=1
print(f'6. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 오일러,베가
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=20*taylor_atan(1/7,i)+8*taylor_atan(3/79,i)
    i+=1
print(f'7. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 크라우젠(틀림)
#start=time()
#result=0
#i=0
#while abs(pi-result)>err:
#    result+=8*taylor_atan(1/5,i)+4*taylor_atan(1/7,i)
#    print(result)
#    i+=1
#print(f'8. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')


# 가우스
start=time()
result=0
i=0

while abs(pi-result)>err:
    result+=12*taylor_atan(1/4,i)+4*taylor_atan(1/20,i)+4*taylor_atan(1/1985,i)
    i+=1
print(f'9. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')
# 다제
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=4*taylor_atan(1/2,i)+4*taylor_atan(1/5,i)+4*taylor_atan(1/8,i)
    i+=1
print(f'10. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 러더퍼드
start=time()
result=0
i=0
while abs(pi-result)>err:
    result+=16*taylor_atan(1/5,i)-4*taylor_atan(1/70,i)+4*taylor_atan(1/99,i)
    i+=1
print(f'11. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 센크스
start=time()
result=0
i=0

while abs(pi-result)>err:
    result+=24*taylor_atan(1/8,i)+8*taylor_atan(1/57,i)+4*taylor_atan(1/239,i)
    i+=1
print(f'12. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 샤프

start=time()
result=0
i=0

while abs(pi-result*(2*sqrt(3)))>err:
    result+=(-1)**i/((2*i+1)*3**i)
    i+=1

result*=2*sqrt(3)    
print(f'13. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 월리스
start=time()
result=1
i=0

while abs(pi-result*2)>err:
    result*=((i//2+1)*2)/(((i+1)//2)*2+1)
    i+=1
result*=2

print(f'14. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')

# 오일러
start=time()
result=2*sqrt(3)
result2=2*sqrt(3)
factor=0
i=0
while abs(pi-result)>err:
    result=result2
    factor+=(-1)**i/(i+1)**2
    result2=result
    result*=sqrt(factor)
    i+=1
print(f'15. 결과: {result}, 걸린 시간: {(time()-start)*1000:.0f}ms, 반복 횟수: {i+1}')
