#include "rotation.h"

void rotate_90_deg(GPIO_TypeDef* GPIO, uint16_t pin0, uint16_t pin1, uint16_t pin2, uint16_t pin3){
	for(int i = 0; i < 128; ++i){
			GPIO_SetBits(GPIO, pin3);
			Delay_ms(2);
			GPIO_ResetBits(GPIO,pin1);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin0);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin2);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin1);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin3);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin2);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin0);
	}
}
void rotate_dot7_deg(GPIO_TypeDef* GPIO, uint16_t pin0, uint16_t pin1, uint16_t pin2, uint16_t pin3){
			GPIO_SetBits(GPIO, pin3);
			Delay_ms(2);
			GPIO_ResetBits(GPIO,pin1);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin0);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin2);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin1);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin3);
			Delay_ms(2);
			GPIO_SetBits(GPIO, pin2);
			Delay_ms(2);
			GPIO_ResetBits(GPIO, pin0);
}
