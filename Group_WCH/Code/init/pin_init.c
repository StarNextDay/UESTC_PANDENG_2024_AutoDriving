#include "stm32f10x.h"                  // Device header

void pin_init(uint16_t GPIO_Pin, GPIOSpeed_TypeDef GPIO_Speed, GPIOMode_TypeDef GPIO_Mode, GPIO_TypeDef* GPIOx)
{
	/*GPIO初始化*/
	GPIO_InitTypeDef GPIO_InitStructure;					//定义结构体变量
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode;		//GPIO模式
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin;				//GPIO引脚
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed;		//GPIO速度
	
	GPIO_Init(GPIOx, &GPIO_InitStructure);					//将赋值后的构体变量传递给GPIO_Init函数
}
