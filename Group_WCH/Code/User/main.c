#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "pin_init.h"
#include "Serial.h"
#include "distance_to_frequency.h"
#include "LED.h"
#include "Bezz.h"

uint32_t RxData;	
int freq=0;
float distance=10000;
float angle=0;


int main(void)
{
	Delay_ms(500);
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	
	//pin_init(GPIO_Pin_12, GPIO_Speed_50MHz, GPIO_Mode_Out_PP, GPIOB);
	Serial_Init();
	
	//蜂鸣器引脚设置
	pin_init(GPIO_Pin_0, GPIO_Speed_50MHz, GPIO_Mode_Out_PP, GPIOA);
	pin_init(GPIO_Pin_1, GPIO_Speed_50MHz, GPIO_Mode_Out_PP, GPIOA);
	pin_init(GPIO_Pin_5, GPIO_Speed_50MHz, GPIO_Mode_Out_PP, GPIOA);
	pin_init(GPIO_Pin_6, GPIO_Speed_50MHz, GPIO_Mode_Out_PP, GPIOA);
	
	//继电器（LED）控制引脚设置（继电器低电平触发）
	pin_init(GPIO_Pin_8, GPIO_Speed_50MHz, GPIO_Mode_Out_OD, GPIOB);
	pin_init(GPIO_Pin_9, GPIO_Speed_50MHz, GPIO_Mode_Out_OD, GPIOB);
	pin_init(GPIO_Pin_10, GPIO_Speed_50MHz, GPIO_Mode_Out_OD, GPIOB);
	pin_init(GPIO_Pin_11, GPIO_Speed_50MHz, GPIO_Mode_Out_OD, GPIOB);
 
	while(1)
	{
		//Serial_SendByte('A');
		if(Serial_GetRxFlag()==1)
		{
			distance=*(float*)&Serial_dis_Data;
			angle=*(float*)&Serial_angle_Data;
			//Serial_SendByte('A');
			//Serial_SendFloat(distance);
			//Serial_SendByte('A');
			//Serial_SendFloat(angle);
		}
		LED_select(distance,angle);
		freq=distance_to_frequency(distance);
		bezz_select(freq, angle);
		
	}
}
