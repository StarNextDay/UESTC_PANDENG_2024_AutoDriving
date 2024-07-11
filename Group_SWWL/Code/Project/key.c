#include "stm32f10x.h"                  // Device header
#include "delay.h"

/**
  * ��    ����������ʼ��
  * ��    ������
  * �� �� ֵ����
  */
void Key_Init(void)
{
	/*����ʱ��*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);		//����GPIOA��ʱ��
	
	/*GPIO��ʼ��*/
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1 | GPIO_Pin_11 | GPIO_Pin_3;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);						//��PA1��PA11���ų�ʼ��Ϊ��������
	
	GPIO_SetBits(GPIOA, GPIO_Pin_3);
}

/**
  * ��    ����������ȡ����
  * ��    ������
  * �� �� ֵ�����°����ļ���ֵ����Χ��0~2������0����û�а�������
  * ע������˺���������ʽ��������������ס����ʱ�������Ῠס��ֱ����������
  */
uint8_t Key_GetNum(void)
{
	uint8_t KeyNum = 0;		//���������Ĭ�ϼ���ֵΪ0
	
	if (GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_1) == 0)			//��PA1����Ĵ�����״̬�����Ϊ0���������1����
	{
		Delay_ms(20);											//��ʱ����
		while (GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_1) == 0);	//�ȴ���������
		Delay_ms(20);											//��ʱ����
		KeyNum = 1;												//�ü���Ϊ1
	}
	
	if (GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_11) == 0)			//��PA11����Ĵ�����״̬�����Ϊ0���������2����
	{
		Delay_ms(20);											//��ʱ����
		while (GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_11) == 0);	//�ȴ���������
		Delay_ms(20);											//��ʱ����
		KeyNum = 2;												//�ü���Ϊ2
	}
	
	return KeyNum;			//���ؼ���ֵ�����û�а������£�����if���������������ΪĬ��ֵ0
}
