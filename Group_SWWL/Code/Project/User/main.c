#include "stm32f10x.h"                  // Device header
#include "delay.h"
#include "rotation.h"
#include "key.h"

// Stepper stepper(STEPS, 12, 13, 14, 15);

int main(void)
{
	Delay_ms(100);
	/*开启时钟*/
	//RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);	//开启GPIOA的时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);	//开启GPIOA的时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);	//开启GPIOC的时钟

//															//使用各个外设前必须开启时钟，否则对外设的操作无效
//	
//	/*GPIO初始化*/
	GPIO_InitTypeDef GPIO_InitStructure;					//定义结构体变量
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;		//GPIO模式，赋值为推挽输出模式
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13;				//GPIO引脚，赋值为第13号引脚
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;		//GPIO速度，赋值为50MHz
	
	// GPIO_Init(GPIOA, &GPIO_InitStructure);					//将赋值后的构体变量传递给GPIO_Init函数
	GPIO_Init(GPIOC, &GPIO_InitStructure);					//将赋值后的构体变量传递给GPIO_Init函数\
	
	//GPIO_InitTypeDef GPIOA_InitStructure;					//定义结构体变量

	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;		//GPIO模式，赋值为推挽输出模式
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15;				//GPIO引脚
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;		//GPIO速度，赋值为50MHz

	GPIO_Init(GPIOB, &GPIO_InitStructure);					//将赋值后的构体变量传递给GPIO_Init函数
	
//															//函数内部会自动根据结构体的参数配置相应寄存器
//															//实现GPIOC的初始化
////	GPIO_SetBits(GPIOB, GPIO_Pin_12);					
////	GPIO_ResetBits(GPIOA, GPIO_Pin_1);					
////	GPIO_ResetBits(GPIOA, GPIO_Pin_2);					
////	GPIO_ResetBits(GPIOA, GPIO_Pin_3);					
////	GPIO_ResetBits(GPIOA, GPIO_Pin_4);					
	GPIO_ResetBits(GPIOC, GPIO_Pin_13);					

	Key_Init();
	/*设置GPIO引脚的高低电平*/
	/*若不设置GPIO引脚的电平，则在GPIO初始化为推挽输出后，指定引脚默认输出低电平*/
	// GPIO_ResetBits(GPIOC, GPIO_Pin_13);						//将PC13引脚设置为低电平
	// GPIO_ResetBits(GPIOA, GPIO_Pin_0);						//将PC13引脚设置为低电平
	
	
//	stepper.setSpeed(200);
//		stepper.step(4096);
	while (1)
	{
		uint8_t key_num = Key_GetNum();
		if(key_num == 1){
			GPIO_ResetBits(GPIOA, GPIO_Pin_3);
			Delay_ms(100);
			GPIO_SetBits(GPIOA, GPIO_Pin_3);
//		for(int i = 0; i < 5; ++i)
//				rotate_dot7_deg(GPIOB, GPIO_Pin_12, GPIO_Pin_13, GPIO_Pin_14, GPIO_Pin_15);
			rotate_90_deg(GPIOB, GPIO_Pin_12, GPIO_Pin_13, GPIO_Pin_14, GPIO_Pin_15);
		}
//		for(int i = 0; i < 10; ++i){
//			rotate_90_deg(GPIOB, GPIO_Pin_12, GPIO_Pin_13, GPIO_Pin_14, GPIO_Pin_15);
//			Delay_ms(100);
//		}
		Delay_ms(200);
		//GPIO_SetBits(GPIOB, GPIO_Pin_12);					//将PA5引脚设置为低电平
//	GPIO_ResetBits(GPIOA, GPIO_Pin_1);					//将PA5引脚设置为低电平
//	GPIO_ResetBits(GPIOA, GPIO_Pin_2);					//将PA5引脚设置为低电平
//	GPIO_ResetBits(GPIOA, GPIO_Pin_3);					//将PA5引脚设置为低电平
//	GPIO_ResetBits(GPIOA, GPIO_Pin_4);					//将PA5引脚设置为低电平
		
// 	GPIO_ResetBits(GPIOB, GPIO_Pin_13);					//将PA5引脚设置为低电平
//		Delay_ms(50);
//		GPIO_ResetBits(GPIOC, GPIO_Pin_13);						//将PC13引脚设置为低电平
//		Delay_ms(500);
//		GPIO_SetBits(GPIOC, GPIO_Pin_13);						//将PC13引脚设置为High电平
//		Delay_ms(500);
	}
}
