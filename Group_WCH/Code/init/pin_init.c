#include "stm32f10x.h"                  // Device header

void pin_init(uint16_t GPIO_Pin, GPIOSpeed_TypeDef GPIO_Speed, GPIOMode_TypeDef GPIO_Mode, GPIO_TypeDef* GPIOx)
{
	/*GPIO��ʼ��*/
	GPIO_InitTypeDef GPIO_InitStructure;					//����ṹ�����
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode;		//GPIOģʽ
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin;				//GPIO����
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed;		//GPIO�ٶ�
	
	GPIO_Init(GPIOx, &GPIO_InitStructure);					//����ֵ��Ĺ���������ݸ�GPIO_Init����
}
