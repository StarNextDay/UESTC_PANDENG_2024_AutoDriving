#include "stm32f10x.h"

// 系统时钟频率，例如72MHz
#define SYSTEM_CLOCK_FREQ 72000000
 
// 用于存储时间戳的变量
extern volatile uint32_t timestamp;
 
// 更新时间戳的中断服务例程
void SysTick_Handler(void);
 
// 初始化SysTick定时器
void SysTick_Init(void);