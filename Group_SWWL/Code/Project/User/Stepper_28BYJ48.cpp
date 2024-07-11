/*
 *
 * The sequence of control signals for 5 phase, 5 control wires is as follows:
 *
 * Step C0 C1 C2 C3 C4
 *    1  0  1  1  0  1
 *    2  0  1  0  0  1
 *    3  0  1  0  1  1
 *    4  0  1  0  1  0
 *    5  1  1  0  1  0
 *    6  1  0  0  1  0
 *    7  1  0  1  1  0
 *    8  1  0  1  0  0
 *    9  1  0  1  0  1
 *   10  0  0  1  0  1
 *
 * The sequence of control signals for 4 control wires is as follows:
 *
 * Step C0 C1 C2 C3
 *    1  1  0  0  0
 *    2  1  1  0  0
 *    3  0  1  0  0
 *    4  0  1  1  0
 *    5  0  0  1  0
 *    6  0  0  1  1
 *    7  0  0  0  1
 *    8  1  0  0  1
 *
 * The sequence of controls signals for 2 control wires is as follows
 * (columns C1 and C2 from above):
 *
 * Step C0 C1
 *    1  0  1
 *    2  1  1
 *    3  1  0
 *    4  0  0
 */

#include "Stepper_28BYJ48.h"
#include "stm32f10x.h"

void init_gpiob() {
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE); // 开启GPIOA的时钟
  GPIO_InitTypeDef GPIO_InitStructure;                 // 定义结构体变量

  GPIO_InitStructure.GPIO_Mode =
      GPIO_Mode_Out_PP; // GPIO模式，赋值为推挽输出模式
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13 | GPIO_Pin_12 |  GPIO_Pin_14 | GPIO_Pin_15; // GPIO引脚，赋值为第0号引脚
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; // GPIO速度，赋值为50MHz

  GPIO_Init(GPIOB, &GPIO_InitStructure);
}
/*
 * two-wire constructor.
 * Sets which wires should control the motor.
 */
Stepper::Stepper(int number_of_steps, int motor_pin_1, int motor_pin_2) {
  this->step_number = 0;    // which step the motor is on
  this->direction = 0;      // motor direction
  this->last_step_time = 0; // time stamp in us of the last step taken
  this->number_of_steps =
      number_of_steps; // total number of steps for this motor

  // Arduino pins for the motor control connection:
  this->motor_pin_1 = motor_pin_1;
  this->motor_pin_2 = motor_pin_2;

  // setup the pins on the microcontroller:
  init_gpiob();
  /* pinMode(this->motor_pin_1, OUTPUT); */
  /* pinMode(this->motor_pin_2, OUTPUT); */

  // When there are only 2 pins, set the others to 0:
  this->motor_pin_3 = 0;
  this->motor_pin_4 = 0;
  this->motor_pin_5 = 0;

  // pin_count is used by the stepMotor() method:
  this->pin_count = 2;
}

/*
 *   constructor for four-pin version
 *   Sets which wires should control the motor.
 */
Stepper::Stepper(int number_of_steps, int motor_pin_1, int motor_pin_2,
                 int motor_pin_3, int motor_pin_4) {
  this->step_number = 0;    // which step the motor is on
  this->direction = 0;      // motor direction
  this->last_step_time = 0; // time stamp in us of the last step taken
  this->number_of_steps =
      number_of_steps; // total number of steps for this motor

  // Arduino pins for the motor control connection:
  this->motor_pin_1 = motor_pin_1;
  this->motor_pin_2 = motor_pin_2;
  this->motor_pin_3 = motor_pin_3;
  this->motor_pin_4 = motor_pin_4;

  // setup the pins on the microcontroller:
  init_gpiob();

  // When there are 4 pins, set the others to 0:
  this->motor_pin_5 = 0;

  // pin_count is used by the stepMotor() method:
  this->pin_count = 4;
}

/*
 *   constructor for five phase motor with five wires
 *   Sets which wires should control the motor.
 */
Stepper::Stepper(int number_of_steps, int motor_pin_1, int motor_pin_2,
                 int motor_pin_3, int motor_pin_4, int motor_pin_5) {
  this->step_number = 0;    // which step the motor is on
  this->direction = 0;      // motor direction
  this->last_step_time = 0; // time stamp in us of the last step taken
  this->number_of_steps =
      number_of_steps; // total number of steps for this motor

  // Arduino pins for the motor control connection:
  this->motor_pin_1 = motor_pin_1;
  this->motor_pin_2 = motor_pin_2;
  this->motor_pin_3 = motor_pin_3;
  this->motor_pin_4 = motor_pin_4;
  this->motor_pin_5 = motor_pin_5;

  // setup the pins on the microcontroller:
  init_gpiob();

  // pin_count is used by the stepMotor() method:
  this->pin_count = 5;
}

/*
 * Sets the speed in revs per minute
 */
void Stepper::setSpeed(long whatSpeed) {
  this->step_delay = 60L * 1000L * 1000L / this->number_of_steps / whatSpeed;
}

/*
 * Moves the motor steps_to_move steps.  If the number is negative,
 * the motor moves in the reverse direction.
 */
void Stepper::step(int steps_to_move) {
  int steps_left = steps_to_move > 0 ? steps_to_move
                                     : -steps_to_move; // how many steps to take

  // determine direction based on whether steps_to_mode is + or -:
  if (steps_to_move > 0) {
    this->direction = 1;
  }
  if (steps_to_move < 0) {
    this->direction = 0;
  }

  // decrement the number of steps, moving one step each time:
  SysTick->CTRL = 0x00000005; // 设置时钟源为HCLK，启动定时器
  SysTick->LOAD = 72 * step_delay * 1000; // 设置定时器重装值
  SysTick->VAL = 0x00;                    // 清空当前计数值
  while (steps_left > 0) {
    // 等到step_delay毫秒后再执行
    while (!(SysTick->CTRL & 0x00010000))
      ;                                     // 等待计数到0
    SysTick->LOAD = 72 * step_delay * 1000; // 设置定时器重装值
    SysTick->VAL = 0x00;                    // 清空当前计数值
    // get the timeStamp of when you stepped:
    /* this->last_step_time = now; */
    // increment or decrement the step number,
    // depending on direction:
    if (this->direction == 1) {
      this->step_number++;
      if (this->step_number == this->number_of_steps) {
        this->step_number = 0;
      }
    } else {
      if (this->step_number == 0) {
        this->step_number = this->number_of_steps;
      }
      this->step_number--;
    }
    // decrement the steps left:
    steps_left--;
    // step the motor to step number 0, 1, ..., {3,8 or 10}
    if (this->pin_count == 5)
      stepMotor(this->step_number % 10);
    else if (this->pin_count == 4)
      stepMotor(this->step_number % 4);
    else
      stepMotor(this->step_number % 4);
  }
  SysTick->CTRL =
      0x00000004; // 关闭定时器/ move only if the appropriate delay has passed:
}

/*
 * Moves the motor forward or backwards.
 */
void Stepper::stepMotor(int thisStep) {
  if (this->pin_count == 2) {
    switch (thisStep) {
    case 0: // 01

      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_SetBits(GPIOB, motor_pin_2);   // 将PA5引脚设置为低电平
      break;
    case 1:                             // 11
      GPIO_SetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_SetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      break;
    case 2:                               // 10
      GPIO_SetBits(GPIOB, motor_pin_1);   // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      break;
    case 3:                               // 00
      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      break;
    }
  }
  if (this->pin_count == 4) {
    switch (thisStep) {
    case 0:                               // 1000
      GPIO_SetBits(GPIOB, motor_pin_1);   // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_3); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_4); // 将PA5引脚设置为低电平
      break;
//    case 1:                               // 1100
//      GPIO_SetBits(GPIOB, motor_pin_1);   // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_2);   // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_3); // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_4); // 将PA5引脚设置为低电平

//      break;
    case 1:                               // 0100
      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_SetBits(GPIOB, motor_pin_2);   // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_3); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_4); // 将PA5引脚设置为低电平
      break;
//    case 3:                               // 0110
//      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_2);   // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_3);   // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_4); // 将PA5引脚设置为低电平
//      break;
    case 2:                               // 0010
      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      GPIO_SetBits(GPIOB, motor_pin_3);   // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_4); // 将PA5引脚设置为低电平
      break;
//    case 5:                               // 0011
//      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_3);   // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_4);   // 将PA5引脚设置为低电平
//      break;
    case 3:                               // 0001
      GPIO_ResetBits(GPIOB, motor_pin_1); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
      GPIO_ResetBits(GPIOB, motor_pin_3); // 将PA5引脚设置为低电平
      GPIO_SetBits(GPIOB, motor_pin_4);   // 将PA5引脚设置为低电平
      break;
//    case 7:                               // 1001
//      GPIO_SetBits(GPIOB, motor_pin_1);   // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_2); // 将PA5引脚设置为低电平
//      GPIO_ResetBits(GPIOB, motor_pin_3); // 将PA5引脚设置为低电平
//      GPIO_SetBits(GPIOB, motor_pin_4);   // 将PA5引脚设置为低电平
//      break;
    }
  }

  /* if (this->pin_count == 5) { */
  /*   switch (thisStep) { */
  /*   case 0: // 01101 */
  /*     digitalWrite(motor_pin_1, LOW); */
  /*     digitalWrite(motor_pin_2, HIGH); */
  /*     digitalWrite(motor_pin_3, HIGH); */
  /*     digitalWrite(motor_pin_4, LOW); */
  /*     digitalWrite(motor_pin_5, HIGH); */
  /*     break; */
  /*   case 1: // 01001 */
  /*     digitalWrite(motor_pin_1, LOW); */
  /*     digitalWrite(motor_pin_2, HIGH); */
  /*     digitalWrite(motor_pin_3, LOW); */
  /*     digitalWrite(motor_pin_4, LOW); */
  /*     digitalWrite(motor_pin_5, HIGH); */
  /*     break; */
  /*   case 2: // 01011 */
  /*     digitalWrite(motor_pin_1, LOW); */
  /*     digitalWrite(motor_pin_2, HIGH); */
  /*     digitalWrite(motor_pin_3, LOW); */
  /*     digitalWrite(motor_pin_4, HIGH); */
  /*     digitalWrite(motor_pin_5, HIGH); */
  /*     break; */
  /*   case 3: // 01010 */
  /*     digitalWrite(motor_pin_1, LOW); */
  /*     digitalWrite(motor_pin_2, HIGH); */
  /*     digitalWrite(motor_pin_3, LOW); */
  /*     digitalWrite(motor_pin_4, HIGH); */
  /*     digitalWrite(motor_pin_5, LOW); */
  /*     break; */
  /*   case 4: // 11010 */
  /*     digitalWrite(motor_pin_1, HIGH); */
  /*     digitalWrite(motor_pin_2, HIGH); */
  /*     digitalWrite(motor_pin_3, LOW); */
  /*     digitalWrite(motor_pin_4, HIGH); */
  /*     digitalWrite(motor_pin_5, LOW); */
  /*     break; */
  /*   case 5: // 10010 */
  /*     digitalWrite(motor_pin_1, HIGH); */
  /*     digitalWrite(motor_pin_2, LOW); */
  /*     digitalWrite(motor_pin_3, LOW); */
  /*     digitalWrite(motor_pin_4, HIGH); */
  /*     digitalWrite(motor_pin_5, LOW); */
  /*     break; */
  /*   case 6: // 10110 */
  /*     digitalWrite(motor_pin_1, HIGH); */
  /*     digitalWrite(motor_pin_2, LOW); */
  /*     digitalWrite(motor_pin_3, HIGH); */
  /*     digitalWrite(motor_pin_4, HIGH); */
  /*     digitalWrite(motor_pin_5, LOW); */
  /*     break; */
  /*   case 7: // 10100 */
  /*     digitalWrite(motor_pin_1, HIGH); */
  /*     digitalWrite(motor_pin_2, LOW); */
  /*     digitalWrite(motor_pin_3, HIGH); */
  /*     digitalWrite(motor_pin_4, LOW); */
  /*     digitalWrite(motor_pin_5, LOW); */
  /*     break; */
  /*   case 8: // 10101 */
  /*     digitalWrite(motor_pin_1, HIGH); */
  /*     digitalWrite(motor_pin_2, LOW); */
  /*     digitalWrite(motor_pin_3, HIGH); */
  /*     digitalWrite(motor_pin_4, LOW); */
  /*     digitalWrite(motor_pin_5, HIGH); */
  /*     break; */
  /*   case 9: // 00101 */
  /*     digitalWrite(motor_pin_1, LOW); */
  /*     digitalWrite(motor_pin_2, LOW); */
  /*     digitalWrite(motor_pin_3, HIGH); */
  /*     digitalWrite(motor_pin_4, LOW); */
  /*     digitalWrite(motor_pin_5, HIGH); */
  /*     break; */
  /*   } */
  /* } */
}

/*
  version() returns the version of the library:
*/
int Stepper::version(void) { return 5; }
