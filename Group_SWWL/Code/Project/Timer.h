#include "stm32f10x.h"

// ϵͳʱ��Ƶ�ʣ�����72MHz
#define SYSTEM_CLOCK_FREQ 72000000
 
// ���ڴ洢ʱ����ı���
extern volatile uint32_t timestamp;
 
// ����ʱ������жϷ�������
void SysTick_Handler(void);
 
// ��ʼ��SysTick��ʱ��
void SysTick_Init(void);