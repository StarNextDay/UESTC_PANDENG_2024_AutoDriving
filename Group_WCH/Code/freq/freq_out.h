#ifndef __FREQ_OUT_H
#define __FREQ_OUT_H

#include "stm32f10x.h"                  // Device header
#include "Delay.h"

void freq_out(int freq, GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
	int time_us=1000000/freq;
	GPIO_WriteBit(GPIOx,GPIO_Pin,Bit_SET);
	Delay_us(time_us);
	GPIO_WriteBit(GPIOx,GPIO_Pin,Bit_RESET);
	Delay_us(time_us);
}

#endif
