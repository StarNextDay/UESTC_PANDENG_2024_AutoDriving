#include "stm32f10x.h"                  // Device header

void LED_select(float distance, float direction)
{
	if(direction>0)
	{
		GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_SET);
		GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_SET);
		if(distance<=80&&distance>40)
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_RESET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_SET);
		}
		else
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_SET);
			if(distance<=40)
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_RESET);
			}
			else
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_SET);
			}
		}
		
	}
	
	else if(direction<0)
	{
		GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_SET);
		GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_SET);
		if(distance<=80&&distance>40)
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_RESET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_SET);
		}
		else
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_SET);
			if(distance<=40)
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_RESET);
			}
			else
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_SET);
			}
		}
		
	}
	
	else
	{
		if(distance<=80&&distance>40)
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_RESET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_SET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_RESET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_SET);
		}
		else
		{
			GPIO_WriteBit(GPIOB,GPIO_Pin_11,Bit_SET);
			GPIO_WriteBit(GPIOB,GPIO_Pin_9,Bit_SET);
			if(distance<=40)
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_RESET);
				GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_RESET);
			}
			else
			{
				GPIO_WriteBit(GPIOB,GPIO_Pin_10,Bit_SET);
				GPIO_WriteBit(GPIOB,GPIO_Pin_8,Bit_SET);
			}
		}
	}
}
