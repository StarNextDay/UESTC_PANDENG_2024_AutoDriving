#ifndef __SERIAL_H
#define __SERIAL_H

#include <stdio.h>

extern uint32_t Serial_dis_Data;		//距离的32位编码
extern uint32_t Serial_angle_Data;		//角度的32位编码

void Serial_Init(void);
void Serial_SendByte(uint8_t Byte);
void Serial_SendArray(uint8_t *Array, uint16_t Length);
void Serial_SendString(char *String);
void Serial_SendNumber(uint32_t Number, uint8_t Length);
void Serial_Printf(char *format, ...);

uint8_t Serial_GetRxFlag(void);
uint8_t Serial_GetRxData(void);

void Serial_SendFloat(float Float);

#endif
