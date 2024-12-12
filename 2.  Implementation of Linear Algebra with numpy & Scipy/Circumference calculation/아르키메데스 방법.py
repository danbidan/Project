import math

n = 5 # 5각형부터 시작
err = 1.0e-7 # 오차

while True:
    degree = 360 / n
    theta = degree / 2
    inner_length = math.sin(math.radians(theta)) * 2 # 내접 정n각형 한 변 길이
    outer_length = math.tan(math.radians(theta)) * 2 # 외접 정n각형 한 변 길이
    difference = outer_length - inner_length
    new_pi = n * ((outer_length + inner_length) / 2) /2 # 아르키메데스 방법으로 구한 파이

    if difference < err: 
        break
    else:
        n = n + 1
        
print(f'pi = {new_pi}')
print(f'정{n}각형')
print("내장 원주율: ", math.pi, "차이: ", math.pi - new_pi)