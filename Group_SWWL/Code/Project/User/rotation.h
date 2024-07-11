#include "stm32f10x.h"
#include "delay.h"

void rotate_90_deg(GPIO_TypeDef* GPIO, uint16_t pin0, uint16_t pin1, uint16_t pin2, uint16_t pin3);
void rotate_dot7_deg(GPIO_TypeDef* GPIO, uint16_t pin0, uint16_t pin1, uint16_t pin2, uint16_t pin3);
