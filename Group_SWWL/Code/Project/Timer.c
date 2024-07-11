#include "stm32f10x.h"

 
// 系统时钟频率，例如72MHz
#define SYSTEM_CLOCK_FREQ 72000000
 
// 用于存储时间戳的变量
volatile uint32_t timestamp = 0;
 
// 更新时间戳的中断服务例程
void SysTick_Handler(void)
{
    timestamp++;
}
 
// 初始化SysTick定时器
void SysTick_Init(void)
{
    // 设置SysTick的加载值
    SysTick->LOAD = (SYSTEM_CLOCK_FREQ / 1000) - 1;
    // 设置SysTick的中断优先级
    NVIC_SetPriority(SysTick_IRQn, (1<<__NVIC_PRIO_BITS) - 1);
    // 使能SysTick定时器中断
    SysTick->CTRL |= SysTick_CTRL_TICKINT_Msk;
    // 使能SysTick定时器
    SysTick->CTRL |= SysTick_CTRL_ENABLE_Msk;
    // 设置SysTick定时器的时钟源为CORECLK
    SysTick->CTRL |= SysTick_CTRL_CLKSOURCE_Msk;
}