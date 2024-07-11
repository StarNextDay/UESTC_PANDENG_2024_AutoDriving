#include "stm32f10x.h"

 
// ϵͳʱ��Ƶ�ʣ�����72MHz
#define SYSTEM_CLOCK_FREQ 72000000
 
// ���ڴ洢ʱ����ı���
volatile uint32_t timestamp = 0;
 
// ����ʱ������жϷ�������
void SysTick_Handler(void)
{
    timestamp++;
}
 
// ��ʼ��SysTick��ʱ��
void SysTick_Init(void)
{
    // ����SysTick�ļ���ֵ
    SysTick->LOAD = (SYSTEM_CLOCK_FREQ / 1000) - 1;
    // ����SysTick���ж����ȼ�
    NVIC_SetPriority(SysTick_IRQn, (1<<__NVIC_PRIO_BITS) - 1);
    // ʹ��SysTick��ʱ���ж�
    SysTick->CTRL |= SysTick_CTRL_TICKINT_Msk;
    // ʹ��SysTick��ʱ��
    SysTick->CTRL |= SysTick_CTRL_ENABLE_Msk;
    // ����SysTick��ʱ����ʱ��ԴΪCORECLK
    SysTick->CTRL |= SysTick_CTRL_CLKSOURCE_Msk;
}