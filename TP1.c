#include<stdio.h>
#include<stdlib.h>

int main()
{
    int x, guess, number;
    printf("do you want 1 or 2");
    scanf("%d",&x);
    if(x==1)
    {
        number = defaut();
        printf("%d",number);
    }
    else if (x==2)
    {
        number = borne();
    }
    while (guess!=number)
    {
        guess = guessNumber();
        printf("you can try again");
    }
}
int defaut()
{
    int x;
    srand(0);
    x = rand();
    return x;
 }
 int borne()
 {
    int x;
    return x;
 }
 int guessNumber()
 {
    int x;
    printf("enter your guess");
    scanf("%d",&x);
    return x;
 }

