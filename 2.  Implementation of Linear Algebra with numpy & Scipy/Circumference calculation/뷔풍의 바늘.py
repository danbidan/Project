import turtle
import random

boardWidth = 40
needleLength = 30
numberOfNeedles = 300
overlappingNeedles = 0
    
myPen = turtle.Turtle()
myPen.hideturtle()
myPen.speed(0)

y = 180
#Draw floor boards
for i in range(0,10):
    myPen.penup()
    myPen.goto(-200,y)
    myPen.pendown()
    myPen.goto(200,y)
    y-=boardWidth

#Draw Needles
myPen.color("#f442d1")
for needle in range(0,numberOfNeedles):
    x=random.randint(-200,200)
    y=random.randint(-180,180)
    angle=random.randint(0,360)
    myPen.penup()
    myPen.goto(x,y)
    myPen.setheading(angle)
    myPen.pendown()
    a=myPen.ycor()
    myPen.forward(needleLength)
    b=myPen.ycor()

    for i in range(-180,181,40):
        if a <= i <= b or b<= i <= a:
            overlappingNeedles+=1
  
print("L = " + str(needleLength))
print("N = " + str(numberOfNeedles))
print("W = " + str(boardWidth))
print("C = "+ str(overlappingNeedles))

pi = (2*needleLength*numberOfNeedles)/(boardWidth*overlappingNeedles)
print("pi ~ ",pi)