import math

n = 10000

s=0
for k in range(n):
    s = s + (1/n) * ((1-(k/n)**2)**(1/2))
new_pi = s*4

print(f'pi = {new_pi}')