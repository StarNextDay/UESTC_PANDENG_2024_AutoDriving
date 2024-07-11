#include "stm32f10x.h"                  // Device header
#include "freq_out.h"


void bezz_select(int freq, float angle)
{
	if(freq!=0)
	{
		if(angle>=-180&&angle<-90)
			freq_out(freq,GPIOA,GPIO_Pin_0);
		else if(angle>=-90&&angle<0)
			freq_out(freq,GPIOA,GPIO_Pin_1);
		else if(angle>=0&&angle<90)
			freq_out(freq,GPIOA,GPIO_Pin_5);
		else if(angle>=90&&angle<180)
			freq_out(freq,GPIOA,GPIO_Pin_6);
	}
}
