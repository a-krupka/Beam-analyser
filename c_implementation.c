#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>

#define n 10

typedef struct load
    {
        float load_Q;
        float pos;
        char type;
        char support_name;
        float lower_q;
        float start_pos;
        float end_pos;
        float length;
        int degree;
    }load;
int pos = 0;
#define my_load loads[pos]
float positions[n];
float hinge_a;
float hinge_b;
float Raz;
float Rbz;

int main()
{
    load loads[n];

    while (1)
    {
        puts("Choose load('S' for Single, 'D' for Distributed,'T' for triangle, 'M' for point moment, 'P' for parabolic load) or get out of loop by typing 'X': ");
        scanf(" %c",&my_load.type);
        my_load.type= toupper(my_load.type);
        if (toupper(my_load.type) == 'X') 
        {
            break;
        }
        switch (my_load.type)
        {
            case 'S':
                puts("Input position: ");
                scanf(" %f",&my_load.pos);
                positions[pos] = my_load.pos;
                puts("Input load: ");
                scanf(" %f",&my_load.load_Q);
                break;
            case 'D':
                puts("Input starting position: ");
                scanf(" %f",&my_load.start_pos);
                puts("Input final position: ");
                scanf(" %f",&my_load.end_pos);
                float dif = my_load.end_pos - my_load.start_pos;
                if (dif <= 0)
                {    
                    puts("wrong input!!!");
                    return 1;
                }
                puts("Input distributed load: ");
                scanf(" %f",&my_load.lower_q);
                my_load.load_Q = dif * my_load.lower_q;
                my_load.pos = dif * 0.5 + my_load.start_pos;
                printf("Load %f, Position %f, Type %c, Start %f, End %f\n",my_load.load_Q,my_load.pos,my_load.type,my_load.start_pos,my_load.end_pos);
                positions[pos] = my_load.start_pos;
                break;
            case 'M':
                puts("Input position: ");
                scanf(" %f",&my_load.pos);
                positions[pos] = my_load.pos;
                puts("Input load: ");
                scanf(" %f",&my_load.load_Q);
                break;
            default:
                puts("Invalid input");
                break;
        }
        printf("Load %f, Position %f, Type %c\n",my_load.load_Q,my_load.pos,my_load.type);
        pos++;
    }
    puts("Input position of point a: ");
    scanf(" %f",&hinge_a);
    my_load.pos = hinge_a;
    my_load.type = 'R';
    pos++;
    printf("počet pos je %i\n",pos);
    puts("Input position of point b: ");
    scanf(" %f",&hinge_b);
    my_load.pos = hinge_b;
    my_load.type = 'R';
    float min_pos = 5000; // potential bug
    for (int i = 0; i <= pos; i++)
    {
        if (loads[i].pos < min_pos)
        {
            min_pos = loads[i].pos;
        }
    }
    printf("Minimal position is %f\n",min_pos);
    int poradi =0;
    if (min_pos != 0)
    {
        for (int i = 0; i <= pos; i++)
        {
            loads[i].pos = loads[i].pos - min_pos;
            printf("pos %i %f ",i,loads[i].pos);
            if (loads[i].type == 'D') // přidat T a P případy
            {
                loads[i].start_pos = loads[i].start_pos - min_pos;
                loads[i].end_pos = loads[i].end_pos - min_pos;
            }
            if (loads[i].type == 'R') // updating positions of hinges
            {
                if (poradi == 0)
                {
                    hinge_a = loads[i].pos;
                    poradi++;
                }
                else
                {
                   hinge_b = loads[i].pos; 
                }
            }
        } 
    }
    float left_side = 0;
    float right_side = 0;
    float moment_sum = 0;
    int bool_mom = 0;
    for (int i = 0; i < pos-1; i++)
    {
        if (loads[i].type != 'M')
        {
            if (hinge_b < loads[i].pos)
            {
                left_side = left_side - (loads[i].load_Q * (loads[i].pos - hinge_b)); 
            }
            else
            {
                left_side = left_side + (loads[i].load_Q * (hinge_b - loads[i].pos));
            }
            
            if (hinge_a < loads[i].pos)
            {
                right_side = right_side - (loads[i].load_Q * (loads[i].pos - hinge_a)); 
            }
            else
            {
                right_side = right_side + (loads[i].load_Q * (hinge_a - loads[i].pos));
            }
        }
        else
        {
            moment_sum += loads[i].load_Q;
            bool_mom = 1;
        }
    }
    printf("Left side %f\n", left_side);
    printf("Right side %f\n", right_side);
    if (bool_mom == 1)
    {
        Raz = (left_side + moment_sum) / (-(hinge_b - hinge_a));
        Rbz = (right_side + moment_sum) / (hinge_b - hinge_a);
    }
    else
    {
        Raz = (left_side) / (-(hinge_b - hinge_a));
        Rbz = (right_side) / (hinge_b - hinge_a);
    }
    printf("Raz = %.2f  Rbz = %.2f \n",Raz,Rbz);    
}
